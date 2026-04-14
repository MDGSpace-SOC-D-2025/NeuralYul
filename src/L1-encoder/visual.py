import sys
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Ignore matplotlib GUI warnings in headless environments
warnings.filterwarnings("ignore", category=UserWarning)

# Import the parser we just built
from parser import YulParser

# ---------------------------------------------------------
# Configuration: NeuralYul Core Types to Colors Mapping
# ---------------------------------------------------------
COLOR_MAP = {
    "BLOCK": "#2C3E50",  # Dark Blue
    "FUNCTION": "#8E44AD",  # Purple
    "OPCODE": "#E74C3C",  # Red
    "VARIABLE": "#27AE60",  # Green
    "LITERAL": "#F39C12",  # Orange
    "OTHER": "#BDC3C7",  # Gray (Should be 0)
}


def visualize_ast(file_path: Path, output_path: Path):
    print(f"[*] Extracting AST and building graph for {file_path.name}...")

    # Generate the graph using your existing parser
    parser = YulParser()
    G = parser.parse(file_path)

    print("[*] Calculating hierarchical tree layout...")

    # pydot crashes if nodes have a 'name' attribute.
    # We create a topology-only copy just for the layout engine.
    try:
        layout_G = nx.DiGraph()
        layout_G.add_nodes_from(G.nodes())
        layout_G.add_edges_from(G.edges())

        pos = nx.nx_pydot.graphviz_layout(layout_G, prog="dot")
    except Exception as e:
        print(f"[!] pydot layout failed: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(G, k=0.15, iterations=20)

    print("[*] Rendering visualization...")
    plt.figure(figsize=(24, 16))

    # Extract node colors based on our architectural types
    node_colors = [
        COLOR_MAP.get(data.get("type", "OTHER")) for _, data in G.nodes(data=True)
    ]

    # Build node labels (e.g., "VARIABLE\n'i'" or "OPCODE\n'add'")
    labels = {}
    for node, data in G.nodes(data=True):
        core_type = data.get("type", "UNKNOWN")
        name = data.get("name", "")
        val = data.get("value", "")

        # Determine the most useful piece of text to show
        identifier = name if name else val
        if identifier:
            labels[node] = f"{core_type}\n{identifier}"
        else:
            labels[node] = f"{core_type}\n({data.get('raw_type', '')})"

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        node_color=node_colors,
        alpha=0.9,
        edgecolors="white",
        linewidths=2,
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="#7F8C8D", arrows=True, arrowsize=20, width=1.5, alpha=0.6
    )
    nx.draw_networkx_labels(
        G, pos, labels, font_size=8, font_weight="bold", font_color="white"
    )

    # Add a custom legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=15,
            label=ntype,
        )
        for ntype, color in COLOR_MAP.items()
    ]
    plt.legend(
        handles=handles,
        loc="upper right",
        title="NeuralYul Node Types",
        fontsize=12,
        title_fontsize=14,
    )

    plt.title(f"NeuralYul SSA-CFG Topology: {file_path.name}", fontsize=20, pad=20)
    plt.axis("off")

    # Save the output
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#ECF0F1")
    print(f"[+] Visualization saved successfully to: {output_path.absolute()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <contract.yul>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[!] File not found: {target}", file=sys.stderr)
        sys.exit(1)

    output_file = target.parent / f"{target.stem}_ast.png"
    visualize_ast(target, output_file)
