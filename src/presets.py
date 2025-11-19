#!/usr/bin/env python3
"""
Voice preset management for maya-say.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Default presets location
PRESETS_DIR = Path.home() / ".config" / "maya"
PRESETS_FILE = PRESETS_DIR / "presets.json"

# Built-in default presets
DEFAULT_PRESETS = {
    "male_american": "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
    "female_american": "Young female voice, clear, american accent, friendly tone, natural pacing.",
    "male_british": "Middle-aged male voice, british accent, formal, clear articulation.",
    "female_british": "Young female voice, british accent, professional, clear delivery.",
    "male_deep": "Deep male voice, commanding presence, slow pacing, authoritative tone.",
    "female_energetic": "Young female voice, excited, energetic, fast pacing, enthusiastic tone.",
    "male_calm": "Middle-aged male voice, calm, soothing, slow pacing, warm timbre.",
    "narrator": "Professional narrator voice, clear articulation, neutral accent, moderate pacing.",
}


class PresetManager:
    """Manages voice presets for maya-say."""

    def __init__(self):
        self.presets_file = PRESETS_FILE
        self.presets = self._load_presets()

    def _load_presets(self) -> Dict[str, str]:
        """Load presets from file, or create with defaults if not exists."""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r') as f:
                    presets = json.load(f)
                    logger.debug(f"Loaded {len(presets)} presets from {self.presets_file}")
                    return presets
            except Exception as e:
                logger.warning(f"Error loading presets: {e}, using defaults")
                return DEFAULT_PRESETS.copy()
        else:
            # Create with defaults
            presets = DEFAULT_PRESETS.copy()
            self._save_presets(presets)
            logger.info(f"Created default presets at {self.presets_file}")
            return presets

    def _save_presets(self, presets: Dict[str, str]) -> None:
        """Save presets to file."""
        self.presets_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.presets_file, 'w') as f:
            json.dump(presets, f, indent=2)
        logger.debug(f"Saved {len(presets)} presets to {self.presets_file}")

    def get_preset(self, name: str) -> Optional[str]:
        """Get a preset by name."""
        return self.presets.get(name)

    def save_preset(self, name: str, description: str) -> None:
        """Save a new preset or update existing."""
        self.presets[name] = description
        self._save_presets(self.presets)
        logger.info(f"Saved preset '{name}'")

    def delete_preset(self, name: str) -> bool:
        """Delete a preset. Returns True if deleted, False if not found."""
        if name in self.presets:
            del self.presets[name]
            self._save_presets(self.presets)
            logger.info(f"Deleted preset '{name}'")
            return True
        return False

    def list_presets(self) -> Dict[str, str]:
        """Get all presets."""
        return self.presets.copy()

    def preset_exists(self, name: str) -> bool:
        """Check if a preset exists."""
        return name in self.presets
