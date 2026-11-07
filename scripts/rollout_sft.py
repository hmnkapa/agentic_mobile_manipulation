#!/usr/bin/env python3
"""Repository entry point for manip-train rollout."""
from pathlib import Path
import sys

# Use this checkout even when launched from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manipulation_agent.training.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["rollout", *sys.argv[1:]]))
