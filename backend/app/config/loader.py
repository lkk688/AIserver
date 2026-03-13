import os
from pathlib import Path
from typing import Any, Dict
from .schema import AppConfig

def load_config(config_path: str = "backend/config.yaml") -> AppConfig:
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load config. "
            "Install it with: conda activate mypy311 && pip install pyyaml"
        )
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
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")

    # Bridge text processing config into environment variables used by the
    # text cleaning utilities. Environment variables (if set) still take
    # precedence over config.yaml via os.environ.get checks.
    tp = config.text_processing
    if tp.kenlm_model_path and not os.getenv("KENLM_MODEL_PATH"):
        os.environ["KENLM_MODEL_PATH"] = str(tp.kenlm_model_path)
    if not os.getenv("APP_TEXT_USE_KENLM"):
        os.environ["APP_TEXT_USE_KENLM"] = "true" if tp.use_kenlm else "false"
    if not os.getenv("APP_TEXT_USE_SEQ2SEQ"):
        os.environ["APP_TEXT_USE_SEQ2SEQ"] = "true" if tp.use_seq2seq else "false"
    if tp.seq2seq_model and not os.getenv("APP_TEXT_SEQ2SEQ_MODEL"):
        os.environ["APP_TEXT_SEQ2SEQ_MODEL"] = tp.seq2seq_model
    if tp.seq2seq_prefix is not None and not os.getenv("APP_TEXT_SEQ2SEQ_PREFIX"):
        os.environ["APP_TEXT_SEQ2SEQ_PREFIX"] = tp.seq2seq_prefix

    return config

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
