from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_all_configs(configs_dir: str = "./configs") -> dict:
    base = Path(configs_dir)
    return {
        "server": load_config(base / "server_config.yaml"),
        "client": load_config(base / "client_config.yaml"),
        "model": load_config(base / "model_config.yaml"),
    }


def get_nested(config: dict, *keys, default=None):
    val = config
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val
