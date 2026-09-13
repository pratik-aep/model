"""Configuration management for iceberg drift prediction."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


class AttrDict(dict):
    """Dict that allows attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


@dataclass
class Config:
    """Main configuration class."""
    paths: Dict[str, str] = field(default_factory=dict)
    data_sources: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert nested dicts to AttrDict for attribute access."""
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = self._convert_to_attrdict(value)

    def _convert_to_attrdict(self, d: Dict) -> AttrDict:
        """Recursively convert dict to AttrDict."""
        result = AttrDict()
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = self._convert_to_attrdict(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split('.')
        value = self.__dict__
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, {})
            elif isinstance(value, AttrDict):
                value = value.get(k, {})
            else:
                return default
        return value if value != {} else default


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or default location."""
    if config_path is None:
        config_path = Path(__file__).parent / "settings.yaml"
    return Config.from_yaml(str(config_path))


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config