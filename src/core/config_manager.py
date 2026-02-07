"""
配置管理器
"""
import json
import os

DEFAULT_CONFIG = {
    "audio": {
        "input_mode": "loopback",
        "device_index": None,
        "sample_rate": 16000
    },
    "translation": {
        "source_lang": "auto",
        "target_lang": "zh-CN"
    },
    "display": {
        "opacity": 0.7,
        "font_size": 24,
        "show_original": True
    },
    "model": {
        "name": "facebook/seamless-m4t-v2-large",
        "device": "cuda",
        "dtype": "float16"
    }
}


class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict:
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    return self._merge_config(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"配置加载失败: {e}")
        return DEFAULT_CONFIG.copy()

    def _merge_config(self, default: dict, loaded: dict) -> dict:
        """递归合并配置"""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"配置保存失败: {e}")

    def get(self, section: str, key: str = None, default=None):
        if key is None:
            return self.config.get(section, default if default is not None else {})
        return self.config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value):
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()
