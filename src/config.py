"""Configuration management for the Better Elo system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration class with fallback to default values."""

    def __init__(self, config_path: str | None = None):
        """Initialize configuration with optional custom path."""
        if config_path is None:
            # Get the project root directory
            project_root = Path(__file__).parent.parent
            config_path = str(project_root / "config" / "default.yaml")

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback defaults."""
        config = {
            "data": {"time_control": "180"},
            "api": {"months": 24},
            "model": {"k_factor": 20, "velocity_window": 10},
            "evolution": {
                "population_size": 1000,
                "generations": 10000,
                "f_value": 0.8,
                "cr_value": 0.9,
                "num_runs": 5,
            },
            "analysis": {
                "stockfish_path": "/usr/bin/stockfish",
                "depth": 20,
                "accuracy_threshold": 50,
            },
        }

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        # Merge with defaults, keeping defaults for missing values
                        for section in config:
                            if section in loaded_config:
                                config[section].update(loaded_config[section])
        except Exception:
            # If loading fails, use defaults
            pass

        return config

    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self._config["data"]

    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration section."""
        return self._config["api"]

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self._config["model"]

    @property
    def evolution(self) -> Dict[str, Any]:
        """Get evolution configuration section."""
        return self._config["evolution"]

    @property
    def analysis(self) -> Dict[str, Any]:
        """Get analysis configuration section."""
        return self._config["analysis"]


# Global configuration instance
config = Config()
