{
  description = "NeuralYul - AI-Driven EVM Gas Optimization Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "neuralyul-dev-env";

          buildInputs = with pkgs; [
            # --- Python ML Stack ---
            python312
            python312Packages.pip
            python312Packages.virtualenv

            # --- Rust Execution Environment ---
            cargo
            rustc
            rustfmt
            clippy
            rust-analyzer
            graphviz

            # --- C++ solc Modding & Verification ---
            cmake
            ninja
            boost           
            z3              
            onnxruntime     
            
            # --- MVP Execution ---
            solc            

            # --- System & Build Utilities ---
            pkg-config
            git
          ]; 

          # --- Environment Variables & Startup Hook ---
          shellHook = ''
            export PROJECT_ROOT="$(pwd)"
            export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
            export RUST_BACKTRACE=1

            if [ ! -d ".venv" ]; then
              echo "Initializing local Python virtual environment (.venv)..."
              python3 -m venv .venv
            fi

            source .venv/bin/activate

            echo "================================================"
            echo " 🧠 NeuralYul Development Environment Active"
            echo "================================================"
            echo " • Python: $(python3 --version | cut -d ' ' -f 2) (venv active)"
            echo " • Rust:   $(rustc --version | cut -d ' ' -f 2)"
            echo " • Z3:     $(z3 --version | grep -o 'version [0-9.]*' | cut -d ' ' -f 2)"
            echo " • solc:   $(solc --version | grep 'Version:' | cut -d ' ' -f 2)"
            echo "================================================"
          '';
        };
      }
    );
}
