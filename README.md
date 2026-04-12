# NeuralYul

> **A Machine-Learning Guided Yul Pass Orchestrator and Bytecode Superoptimizer for the Solidity Compiler**

Every smart contract deployed on Ethereum has a financial cost attached to every operation it executes. The standard Solidity compiler (`solc`) attempts to reduce these gas costs by running a fixed, hardcoded sequence of transformation passes on its intermediate representation, Yul.

NeuralYul replaces this static heuristic with a reinforcement learning architecture. By embedding the compiler's intermediate representation into a continuous vector space and orchestrating passes dynamically, NeuralYul minimizes execution gas or zk-proof constraints with mathematical certainty.

---

## 🧠 System Architecture

NeuralYul operates across two distinct optimization boundaries to bridge the semantic gap between high-level logic and low-level machine constraints.

### 1. Yul-MLGO (The Strategist)

An Ahead-of-Time (AOT) compiler orchestrator embedded directly inside a modified `solc` binary.

- **State Representation:** Converts the Yul AST into a Heterogeneous Graph Attention Network (GAT) embedding combined with a Transformer encoder layer to capture both local data-flow and global state dependencies.
- **Policy Orchestration:** A Proximal Policy Optimization (PPO) agent predicts the optimal sequence of the compiler's 24 built-in transformation passes (e.g., `DeadCodeEliminator`, `FullInliner`) based on the specific contract topology.
- **Pre-training:** The graph encoder is pre-trained on the YulCode dataset (350,000 contracts) for gas regression and pass applicability classification.

### 2. Bytecode Superoptimizer (The Tactician)

A post-compilation reinforcement learning loop that directly mutates raw EVM opcodes to squeeze out micro-gas inefficiencies.

- **Concolic Execution:** Resolves dynamic `JUMP` and `JUMPI` destinations using abstract interpretation and concrete fuzzing traces to reconstruct a deterministic Control Flow Graph.
- **Constrained MDP (CMDP):** The RL environment is formulated using Lagrangian duality to enforce mathematical safety. The agent maximizes gas savings while a strict cost function penalizes semantic divergence.
- **Objective Function:** $\max_\theta \min_{\lambda \geq 0} \mathcal{L}(\theta, \lambda) = J_R(\pi_\theta) - \lambda \cdot (J_C(\pi_\theta) - \epsilon)$
- **Hardware Enforcement:** The EVM's strict 16-slot stack limit is physically enforced during RL exploration via dynamic action masking.

### 3. The Correctness Gate

No optimized contract leaves the system without passing strict verification.

- **Fast Inner Loop:** Differential fuzzing validates output, storage state, and gas reduction across thousands of generated calldata inputs.
- **Formal Verification (Z3):** The final hyper-optimized bytecode is mathematically proven against the original code using the Z3 Theorem Prover.
- **UIF Abstraction:** `KECCAK256` state space explosions are bypassed using Uninterpreted Functions (UIFs), proving semantic equivalence via mathematical congruence.

---

## ⚡ Core Features

- **Zero-Copy Rust Execution Environment:** Evaluates the RL reward function by wrapping `revm` (the fastest Rust EVM) in a shared memory ring buffer. This bypasses Python-Rust FFI serialization bottlenecks to achieve ~100,000 state transitions per second.
- **Dual-Target Cost Functions:** Compiles optimally for either Ethereum Mainnet (penalizing `SSTORE`) or zkEVM rollups (penalizing `KECCAK256` and optimizing for PLONK constraint reduction).
- **Multi-Agent RL (MARL):** For large DeFi protocols, NeuralYul assigns independent agents to separate Yul functions under a single coordinator, utilizing leave-one-out attribution for accurate credit assignment.

---

## 🛠 Developer Experience (DevEx) & Usage

NeuralYul hooks directly into standard development frameworks via a lightweight Rust wrapper, requiring no new security assumptions or workflow changes.

**Fast Development Loop (Seconds)**
Use the embedded C++ Yul-MLGO model for rapid, phase-ordered structural optimization.

```bash
forge build --ml-optimize
```
