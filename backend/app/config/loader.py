import os
import yaml
from pathlib import Path
from typing import Any, Dict
from .schema import AppConfig

def load_config(config_path: str = "backend/config.yaml") -> AppConfig:
    # Allow absolute paths or paths relative to CWD
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / config_path
        
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.absolute()}")

    with open(path, "r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")

    # Apply env var overrides (basic implementation)
    # Convention: APP_SECTION_FIELD (e.g., APP_STORAGE_DATA_DIR)
    _apply_env_overrides(raw_config)

    try:
        config = AppConfig(**raw_config)
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

def _apply_env_overrides(config: Dict[str, Any], prefix: str = "APP"):
    """
    Recursively apply environment variables to the config dictionary.
    """
    for key, value in config.items():
        env_key = f"{prefix}_{key.upper()}"
        if isinstance(value, dict):
            _apply_env_overrides(value, env_key)
        else:
            env_val = os.getenv(env_key)
            if env_val is not None:
                # Type casting could be improved here, but strictly relying on Pydantic validation later
                # For basic types like int/bool, we might need simple conversion
                if isinstance(value, bool):
                     config[key] = env_val.lower() in ("true", "1", "yes")
                elif isinstance(value, int):
                    try:
                        config[key] = int(env_val)
                    except ValueError:
                        pass # Ignore invalid env override types, let Pydantic catch it
                else:
                    config[key] = env_val
