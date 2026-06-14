# Technical Specification: Architectural Migration to Graph Contrastive Learning

**Version:** 2.0

---

## 1. Executive Summary

The baseline NeuralYul encoder architecture uses a Graph Convolutional Network (GCN) trained via Supervised Learning to predict a scalar target (static gas cost). While sufficient for pipeline verification, this approach suffers from severe **Many-to-One Mapping (Embedding Collapse)** — topologically distinct Yul program graphs that happen to produce identical scalar gas values are forced to the same point in latent space. This blinds the downstream Reinforcement Learning (RL) agent to critical control-flow and data-dependency nuances that are essential for finding non-trivial optimization strategies.

This document specifies the migration to a **Self-Supervised Graph Contrastive Learning Architecture** (drawing from SimCLR, BGRL, and Joint Embedding Predictive Architecture / JEPA design principles). By eliminating human-defined scalar targets and training the network to maximize mutual information between structurally mutated views of the same underlying program graph, we force the GNN to learn the structural semantics of Yul code entirely from its own topology.

The architecture is split into two phases:

- **Phase 1 — Self-Supervised Pre-training:** The GNN encoder and a temporary projection head are trained with NT-Xent contrastive loss on augmented program dependence graphs. No labels are used.
- **Phase 2 — RL Fine-Tuning:** The projection head is discarded. The frozen GNN backbone acts as a structural feature extractor feeding the RL agent's policy network.

---

## 2. Architecture Overview

### 2.1 Pipeline Comparison

```
Current (Supervised Baseline):
[Raw Yul]
   └─> [AST Parser]
         └─> [GNN Encoder (3-layer GCN)]
               └─> [Global Mean Pool]
                     └─> [Linear Head]
                           └─> MSE Loss vs. scalar gas cost

Proposed (Self-Supervised Contrastive — GraphCLR):
[Raw Yul]
   └─> [PDG Parser — AST + CFG + DFG edges]
         └─> [Augmentation View t1]──> [GNN Backbone f_θ]──> [Pool]──> [Projector g_φ]──┐
         └─> [Augmentation View t2]──> [GNN Backbone f_θ]──> [Pool]──> [Projector g_φ]──┴──> NT-Xent Loss
```

### 2.2 Component Comparison Table

| Component              | Baseline Architecture                              | Proposed Architecture (GraphCLR)                                                 |
| ---------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Parser Target**      | Abstract Syntax Tree (AST) parent-child edges only | Program Dependence Graph: AST + Control Flow Graph (CFG) + Data Flow Graph (DFG) |
| **Supervision Signal** | Scalar label (`target_gas`)                        | None — fully self-supervised                                                     |
| **Loss Function**      | Mean Squared Error (L_MSE)                         | Normalized Temperature-scaled Cross Entropy (L_NT-Xent)                          |
| **Pooling Strategy**   | Global Mean Pool (single readout)                  | Concatenated Global Add + Global Max Pool (dual readout)                         |
| **Negative Sampling**  | N/A                                                | In-batch negatives, minimum batch size N=512                                     |
| **Latent Space**       | Collapsed — scalar gas proxy                       | Expressive — clustered by topological and behavioral similarity                  |
| **Temperature**        | N/A                                                | Learnable τ, initialized at 0.2                                                  |
| **Momentum Encoder**   | N/A                                                | EMA momentum encoder on one branch (m=0.996) to prevent collapse                 |
| **Compute Overhead**   | Very Low                                           | High — two augmented views per graph, doubled forward pass per step              |

---

## 3. Core Mathematical Formulation

### 3.1 Contrastive Objective

Given a batch of N Yul program graphs `{G_1, G_2, ..., G_N}`, two stochastic augmentations `t1` and `t2` are independently sampled for each graph, producing 2N augmented views. The encoder `f_θ` maps each view to an embedding `h`, and the projection head `g_φ` maps `h` to a normalized vector `z` in the contrastive space.

The NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for a positive pair `(i, j)` derived from the same source graph is:

```
ℓ(i,j) = −log [ exp(sim(z_i, z_j) / τ) / Σ_{k=1}^{2N} 𝟙[k≠i] · exp(sim(z_i, z_k) / τ) ]
```

The total loss averages over all 2N positive pairs in the batch:

```
L_NT-Xent = (1 / 2N) · Σ_{k=1}^{N} [ ℓ(2k−1, 2k) + ℓ(2k, 2k−1) ]
```

Where:

- `z_i`, `z_j` are L2-normalized projected latent vectors from the projection head `g_φ`
- `sim(u, v)` is cosine similarity: `(u · v) / (||u|| · ||v||)`
- `τ` is a **learnable** temperature parameter, initialized at `0.2`
- `N` is the batch size (minimum 512 graphs)
- `𝟙[k≠i]` is an indicator function excluding the anchor from the denominator

### 3.2 Temperature as a Learnable Parameter

A fixed τ = 0.07 — the SimCLR ImageNet default — is calibrated for dense, high-dimensional image patch similarity. Yul program graphs are sparse, structurally heterogeneous, and exhibit a much wider variance in pairwise similarity distribution. A τ this low makes the softmax denominator numerically unstable on graph data and causes gradient explosion in early training epochs. Temperature is therefore implemented as a learnable `nn.Parameter` with a floor clamp at 0.05 so it adapts to the actual geometry of the embedding space during training.

### 3.3 Momentum Encoder (Asymmetric Branch)

To prevent representational collapse without requiring prohibitively large batch sizes, one branch uses an Exponential Moving Average (EMA) encoder `f_ξ` whose weights are never directly updated by gradients:

```
ξ ← m · ξ + (1 − m) · θ
```

Where `m = 0.996` is the momentum coefficient. Only the online branch `f_θ` receives gradient updates. This asymmetric design is the core stability mechanism from BYOL and is essential for datasets smaller than approximately 100k graphs.

---

## 4. Program Dependence Graph (PDG) Parser

### 4.1 Edge Type Taxonomy

The PDG parser must extract and label three distinct edge types. Contrastive augmentations operate selectively on them, so correct labeling at parse time is a hard requirement.

| Edge Type          | Label | Semantic Meaning                                    | Augmentation Target            |
| ------------------ | ----- | --------------------------------------------------- | ------------------------------ |
| `AST_PARENT_CHILD` | 0     | Syntactic tree structure                            | **Protected — never dropped**  |
| `CONTROL_FLOW`     | 1     | Jump/branch targets between basic blocks            | Droppable (10% rate)           |
| `DATA_FLOW`        | 2     | Variable definitions reaching uses (def-use chains) | Primary drop target (15% rate) |

Structural edges (`AST_PARENT_CHILD`) must never be perturbed. Dropping them can produce syntactically invalid graph views that correspond to no valid Yul program, making the contrastive task incoherent.

### 4.2 Node Feature Vector

Each node is encoded as a fixed-length 36-dimensional feature vector.

| Dimensions | Feature              | Encoding                                                         |
| ---------- | -------------------- | ---------------------------------------------------------------- |
| 0–31       | Opcode / Node Type   | 32-class one-hot covering the full Yul IR node vocabulary        |
| 32–33      | Edge type flags      | Binary indicators: `has_data_flow_edge`, `has_control_flow_edge` |
| 34         | Stack depth estimate | Normalized integer in `[0, 1]`                                   |
| 35         | Is function boundary | Binary                                                           |

**Total input dimension: 36.** This value must be used as `input_dim` in all model constructors.

---

## 5. Graph Augmentations for Yul Code

All three augmentations are compiler-aware — they preserve semantic validity of the graph views. Each call to `dataset.get(idx)` independently samples a different combination of two augmentations for view `t1` and view `t2`. The two views must always use different augmentation combinations.

### 5.1 Augmentation 1 — Node Feature Masking (NFM)

Randomly zeros out the feature vector of 15% of nodes selected uniformly. Forces the GNN to infer node identity from structural neighborhood context rather than memorizing feature patterns.

### 5.2 Augmentation 2 — Edge Perturbation (EP)

Randomly drops `DATA_FLOW` edges (label 2) at a 15% rate and `CONTROL_FLOW` edges (label 1) at a 10% rate. `AST_PARENT_CHILD` edges are never dropped. If all edges of a given type are removed (possible in small graphs), the original edge set for that type is restored from the pre-drop state of the augmented copy — not from the pristine original graph — to maintain consistency with any NFM mutations already applied.

### 5.3 Augmentation 3 — Subgraph Extraction (SE)

Extracts the k-hop neighborhood subgraph (k=3) centered on a randomly selected basic block entry node. This simulates a local optimization view of the program and is the primary source of augmentation diversity for large contracts. SE is only applied to graphs with more than 20 nodes; smaller graphs skip SE and fall back to NFM+EP only. The extracted subgraph must re-index node IDs locally and carry over the original node feature vectors.

---

## 6. Model Architecture Specification

### 6.1 GNN Backbone `f_θ`

The backbone uses three Graph Isomorphism Network (GIN) convolution layers. GCN's symmetric normalization (`D^{-1/2} A D^{-1/2}`) is designed for homophilic node classification and is provably less expressive than GIN for distinguishing non-isomorphic graphs (Xu et al., 2019). For a task that fundamentally depends on structural discrimination between program graphs, GIN is the correct architecture.

| Layer   | Type                              | Input Dim | Output Dim         | Notes                                    |
| ------- | --------------------------------- | --------- | ------------------ | ---------------------------------------- |
| `conv1` | GINConv (2-layer MLP)             | 36        | 128                | BatchNorm + ReLU                         |
| `conv2` | GINConv (2-layer MLP)             | 128       | 256                | BatchNorm + ReLU                         |
| `conv3` | GINConv (2-layer MLP)             | 256       | 256                | BatchNorm + ReLU                         |
| Readout | Global Add Pool + Global Max Pool | 256       | 512 (concatenated) | Dual readout — replaces single mean pool |
| `norm`  | LayerNorm                         | 512       | 512                | Stabilizes embedding magnitudes          |

**Total backbone parameters (f_θ): ~1.4M**

Single Global Mean Pool is not used as the sole readout because it can produce identical output vectors for structurally different graphs with similar node-count distributions. The dual Add+Max concatenation preserves both aggregate magnitude and peak activation signals.

### 6.2 Projection Head `g_φ`

The projection head maps the 512-dim backbone output to a lower-dimensional contrastive space. It is used only during pre-training and is discarded before RL fine-tuning. The RL agent receives the 512-dim backbone embedding directly from `f_θ`.

| Layer | Type                      | Input Dim | Output Dim | Notes                                      |
| ----- | ------------------------- | --------- | ---------- | ------------------------------------------ |
| `fc1` | Linear + BatchNorm + ReLU | 512       | 256        |                                            |
| `fc2` | Linear + BatchNorm + ReLU | 256       | 128        |                                            |
| `fc3` | Linear                    | 128       | 64         | No activation — L2 norm applied externally |

**Projection head parameters: ~200K**  
**Total trainable parameters (online branch): ~1.6M**  
**Total parameters including EMA target branch: ~3.2M** (EMA branch carries no gradient state)

### 6.3 Backbone Checkpoint Strategy

After pre-training, the checkpoint save procedure must explicitly write the backbone state dict (`f_θ`) and the projection head state dict (`g_φ`) to separate keys. The RL agent loads only the backbone key. Saving the full model as a single blob and expecting downstream users to manually slice the state dict is not acceptable — it creates weight key mismatches when the projection head layers are absent from the agent's model definition.

---

## 7. Dataset Engine Specification

### 7.1 Parent Class

`torch_geometric.data.InMemoryDataset` is the correct parent class for in-memory graph lists. Using `torch_geometric.data.Dataset` requires a `root` directory for on-disk caching and raises a `TypeError` at instantiation when no path is provided.

### 7.2 Augmentation Consistency Requirement

When edge perturbation drops all edges of a given type, the fallback must restore from the augmented copy's pre-drop state for that edge type only — not from the original unaugmented `data` object. Restoring from the original after NFM has already been applied creates an inconsistent view where node features reflect one mutation state while edge connectivity reflects another.

### 7.3 Subgraph Extraction Dependencies

SE requires `torch_geometric.utils.k_hop_subgraph`. Graphs with fewer than 20 nodes skip SE silently and emit a warning to the dataset statistics tracker for monitoring during data loading.

---

## 8. Training Configuration & Hyperparameters

### 8.1 Full Hyperparameter Specification

| Hyperparameter                | Value                         | Notes                                                                                |
| ----------------------------- | ----------------------------- | ------------------------------------------------------------------------------------ |
| `learning_rate`               | 1e-3 (warmup) → 3e-4 (steady) | Cosine decay with 10-epoch linear warmup                                             |
| `optimizer`                   | AdamW                         | Weight decay 1e-4                                                                    |
| `batch_size`                  | 512                           | Hard minimum; 256 + gradient accumulation ×2 is the constrained-hardware alternative |
| `epochs` (pre-training)       | 100–200                       | Early stop on SVD entropy plateau, patience=10                                       |
| `temperature τ_init`          | 0.2                           | Learnable `nn.Parameter`; clamped to [0.05, 0.5]                                     |
| `momentum m`                  | 0.996                         | EMA update coefficient for target encoder                                            |
| `NFM mask probability`        | 0.15                          | 15% of nodes per view                                                                |
| `EP drop rate (DATA_FLOW)`    | 0.15                          |                                                                                      |
| `EP drop rate (CONTROL_FLOW)` | 0.10                          |                                                                                      |
| `SE k-hop depth`              | 3                             | Only on graphs with >20 nodes                                                        |
| `GIN epsilon`                 | 0.0 (learned)                 | Standard GIN configuration                                                           |
| `dropout`                     | 0.1                           | Applied between GIN layers only                                                      |
| `gradient clip norm`          | 1.0                           | Prevents instability from early contrastive loss spikes                              |
| `input_dim`                   | 36                            | Full node feature vector                                                             |
| `embedding_dim`               | 512                           | Post dual-pool concatenation                                                         |
| `projection_dim`              | 64                            | Output of `g_φ`                                                                      |

### 8.2 Minimum Batch Size Justification

NT-Xent depends entirely on in-batch negatives. For a batch of N graphs, each anchor has 2N−2 negatives available. At N=512 this gives 1,022 negatives per anchor — the practical floor for meaningful contrastive signal on a vocabulary-rich domain like Yul opcodes. At N=128 (256 negatives), the loss surface is too coarse for the model to learn fine-grained structural distinctions. Batch sizes below 256 are not supported.

---

## 9. Hardware & Compute Requirements

### 9.1 Phase 1 — Self-Supervised Pre-training

**VRAM analysis per forward pass at batch N=512:**

- Each batch holds 2N = 1,024 augmented graphs simultaneously (online + target branch)
- Average Yul graph: ~150 nodes, ~200 edges, 36-dim features
- Per-graph tensor memory: ~150 × 36 × 4 bytes ≈ 21 KB
- Raw batch tensor memory: ~21 KB × 1,024 ≈ 21 MB
- GIN intermediate activations (3 layers, 128/256/256 dims): ~800 MB at batch 512
- EMA encoder duplicate forward pass: ~400 MB additional
- Optimizer states (AdamW, fp32 master weights): ~25 MB
- **Total VRAM required: 14–18 GB under mixed precision (fp16 activations, fp32 master weights)**

**Realistic hardware tiers:**

| Tier               | Hardware                                        | Effective Batch Size              | Estimated Time (100 epochs) | Notes                                                          |
| ------------------ | ----------------------------------------------- | --------------------------------- | --------------------------- | -------------------------------------------------------------- |
| **Minimum viable** | RTX 3090 24GB / RTX 4090 24GB                   | 512                               | 36–60 hours                 | Requires gradient checkpointing + mixed precision              |
| **Recommended**    | 2× RTX 4090 (48 GB total) / A100 40 GB          | 512–1024                          | 18–30 hours                 | DDP training; preferred for iteration speed                    |
| **Cloud**          | A100 80 GB SXM                                  | 1024–2048                         | 10–18 hours                 | Larger batch = stronger contrastive signal                     |
| **Apple Silicon**  | M3 Max 128 GB unified memory                    | 256 with gradient accumulation ×2 | 72–96 hours                 | Feasible but slow; MPS backend has incomplete PyG support      |
| **Not viable**     | RTX 3060 12 GB / RTX 3070 8 GB / any GPU <16 GB | —                                 | OOM                         | Cannot hold both branches of the EMA architecture at batch 512 |

### 9.2 Phase 2 — RL Fine-Tuning

The CPU is the bottleneck in Phase 2. The frozen GNN runs inference only; GPU load is negligible.

| Resource      | Minimum                                 | Recommended                                              | Notes                                                                                                  |
| ------------- | --------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **CPU cores** | 16 physical cores (e.g., Ryzen 9 5950X) | 32–64 cores (Threadripper 3970X, EPYC, dual-socket Xeon) | Each vectorized env spawns a `solc` subprocess; 16 cores limits parallelism to 8–12 envs               |
| **RAM**       | 64 GB                                   | 128 GB                                                   | PDG-parsed dataset in PyG format for ~50k contracts occupies 40–50 GB; leaves headroom for env workers |
| **Storage**   | 200 GB NVMe SSD                         | 500 GB NVMe SSD                                          | Parsed PDG cache; spinning HDDs cause DataLoader starvation                                            |
| **GPU**       | Any modern GPU                          | Any modern GPU                                           | GNN is frozen; RL policy network is small; GPU is largely idle                                         |

### 9.3 Memory Optimization Strategies for Single 24 GB GPU

1. **Gradient checkpointing** on GIN layers — trades approximately 30% runtime speed for ~40% reduction in activation memory
2. **Mixed precision training** via `torch.amp.autocast` with fp16 activations and fp32 master weights — reduces activation memory by ~50%
3. **Gradient accumulation** — accumulate over 4 steps at batch 128 to simulate an effective batch of 512 without materializing all 1,024 augmented graphs simultaneously
4. **Sparse tensor representations** — store edge indices as COO sparse tensors; never materialize dense adjacency matrices at any point in the pipeline

---

## 10. Verification Metrics

### 10.1 Embedding Collapse Detection — SVD Entropy

After each validation epoch, compute the singular value decomposition of the embedding matrix `Z ∈ R^{M×512}` over a fixed held-out validation set of M=1,000 graphs. The effective rank entropy is:

```
H_svd = − Σ_i p_i · log(p_i),   where p_i = σ_i / Σ_j σ_j
```

A collapsing model concentrates all variance in 1–3 singular values, driving H_svd toward zero. A healthy model should have H_svd > 4.0 by epoch 20. If H_svd < 2.0 after 30 epochs, training has collapsed and should be restarted with a higher τ_init or reduced augmentation drop rates.

### 10.2 Functional Clustering — t-SNE Validation

A 2D t-SNE projection of extracted embeddings over the validation set must show functional grouping: programs with similar control-flow patterns (loops, conditional branches, linear sequences) must cluster together regardless of their gas cost.

The primary qualitative benchmark is the **structural clone test**: construct a held-out set of pairs of Yul contracts that produce identical gas costs but use different loop architectures (e.g., a `for` loop vs. an unrolled equivalent). In the baseline supervised model, these collapse to the same point. In a correctly trained contrastive model, they must appear as distinct clusters separated in the t-SNE projection.

### 10.3 Downstream RL Reward Velocity

The ultimate quantitative benchmark. After connecting the frozen GNN encoder to the RL agent and beginning Phase 2 training, the contrastive GIN encoder must show faster reward velocity than the supervised GCN baseline in the first 50k RL steps. The agent can now distinguish structurally different programs that yield the same gas cost, enabling it to find non-trivial optimization paths that the baseline cannot perceive.

A statistically significant improvement in mean reward at 50k steps (p < 0.05 over 3 independent seeds) constitutes final validation of the architecture migration.

---
