"""Configuration loader utility."""
import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize config loader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Handle relative paths - find project root
        if not os.path.isabs(config_path):
            # Try to find project root by looking for config directory
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            config_path = project_root / config_path
            
            # If not found, try current working directory
            if not config_path.exists():
                config_path = Path(config_path).resolve()
        
        self.config_path = str(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports nested keys with dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config

