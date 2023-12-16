from pathlib import Path
import json

with open(Path(__file__).resolve().parent / "config.json", 'r') as f:
    config = json.load(f)
