import hashlib
import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)


def run_id_from_config(cfg: dict[str, Any], length: int = 8) -> str:
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
