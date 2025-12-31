
import hashlib
import json
from pathlib import Path
from typing import Any

# Default path relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "paper_run.json"

class Config:
    def __init__(self, data: dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}")
        with path.open("r") as f:
            data = json.load(f)
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Returns a deep copy of the configuration dictionary."""
        return json.loads(json.dumps(self._data))

    def resolve(self) -> dict[str, Any]:
        """
        Returns the configuration dictionary. 
        In strict mode, this would resolve any dynamic defaults.
        For now, it returns the loaded dict as the single source of truth.
        """
        return self.to_dict()

    def get_run_id(self, length: int = 8) -> str:
        """Generates a deterministic ID based on the config content."""
        canonical = json.dumps(self._data, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:length]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

# Backwards compatibility helper (if needed, but we should refactor usages)
def load_config(path: str | Path) -> dict[str, Any]:
    return Config.load(path).to_dict()

def run_id_from_config(cfg: dict[str, Any], length: int = 8) -> str:
    # Re-wrap just to use the method
    return Config(cfg).get_run_id(length)
