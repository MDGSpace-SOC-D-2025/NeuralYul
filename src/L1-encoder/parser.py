import sys
import json
import subprocess
from pathlib import Path
import networkx as nx


class YulParser:
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_id = 0

    def get_ast(self, file_path: Path) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            yul_code = f.read()

        payload = {
            "language": "Yul",
            "sources": {file_path.name: {"content": yul_code}},
            "settings": {
                "outputSelection": {"*": {"": ["ast"], "*": ["*"]}},
                "optimizer": {"enabled": True},
            },
        }

        try:
            res = subprocess.run(
                ["solc", "--standard-json"],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                check=True,
            )
            output = json.loads(res.stdout)

            if "sources" not in output or file_path.name not in output["sources"]:
                print(
                    f"[!] solc withheld the AST. Raw compiler response:",
                    file=sys.stderr,
                )
                print(json.dumps(output, indent=2), file=sys.stderr)
                sys.exit(1)

            return output["sources"][file_path.name]["ast"]

        except Exception as e:
            print(f"[!] Failed to execute solc:\n{e}", file=sys.stderr)
            sys.exit(1)

    def _get_core_type(self, raw_type: str) -> str:
        """Map raw solc Yul types to our 5 architectural core types."""
        raw = raw_type.lower()

        # Operations and execution steps
        if (
            "functioncall" in raw
            or ("identifier" in raw and "builtin" in raw)
            or "assignment" in raw
        ):
            return "OPCODE"

        # Data and memory allocations
        if "variabledeclaration" in raw or "identifier" in raw or "typedname" in raw:
            return "VARIABLE"

        # Constants
        if "literal" in raw:
            return "LITERAL"

        # Scope boundaries, control structures, and namespaces
        if any(
            b in raw
            for b in [
                "block",
                "object",
                "code",
                "loop",
                "switch",
                "case",
                "expressionstatement",
            ]
        ):
            return "BLOCK"

        # Callable boundaries
        if "functiondefinition" in raw:
            return "FUNCTION"

        return "OTHER"

    def _traverse(self, node: dict, parent_id: int = None) -> int:
        """Generically walks the AST dict and populates the NetworkX graph."""
        current = self.node_id
        self.node_id += 1

        raw_type = node.get("nodeType", "Unknown")

        self.G.add_node(
            current,
            type=self._get_core_type(raw_type),
            raw_type=raw_type,
            name=node.get("name", ""),
            value=node.get("value", ""),
        )

        if parent_id is not None:
            self.G.add_edge(parent_id, current, edge_type="AST_PARENT_CHILD")

        # Recursively search every key for nested AST nodes
        for key, value in node.items():
            # Skip metadata keys to prevent infinite loops or false positives
            if key in ("nodeType", "name", "value", "src", "nativeSrc"):
                continue

            # If the value is a dict and has a nodeType, it's a child node
            if isinstance(value, dict) and "nodeType" in value:
                self._traverse(value, current)

            # If the value is a list, check its items for child nodes
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "nodeType" in item:
                        self._traverse(item, current)

        return current

    def parse(self, file_path: Path) -> nx.DiGraph:
        print(f"[*] Parsing {file_path.name}...")
        ast = self.get_ast(file_path)

        print("[*] Building SSA-CFG...")
        self._traverse(ast)

        print(
            f"[+] Done. Nodes: {self.G.number_of_nodes()} | Edges: {self.G.number_of_edges()}"
        )
        return self.G


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <contract.yul>", file=sys.stderr)
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"[!] File not found: {target}", file=sys.stderr)
        sys.exit(1)

    parser = YulParser()
    G = parser.parse(target)

    counts = {}
    raw_others = set()
    for _, data in G.nodes(data=True):
        t = data["type"]
        counts[t] = counts.get(t, 0) + 1
        if t == "OTHER":
            raw_others.add(data["raw_type"])

    print("\nTopology Breakdown:")
    for node_type, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type:<10} : {count}")

    if raw_others:
        print(f"\nUnmapped 'OTHER' raw types: {', '.join(raw_others)}")
