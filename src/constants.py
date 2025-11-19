#!/usr/bin/env python3
"""
Constants and configuration for maya-say TTS system.
"""

from pathlib import Path

# Model configuration
MODEL_NAME = "maya-research/maya1"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"

# Token IDs for Maya1
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259  # Start of Header
EOH_ID = 128260  # End of Header
SOA_ID = 128261  # Start of Audio
BOS_ID = 128000  # Beginning of Sequence
TEXT_EOT_ID = 128009  # End of Text

# Audio configuration
SAMPLE_RATE = 24000
WARMUP_SAMPLES = 2048  # Samples to trim from start of audio (~0.085s as per official example)

# Server configuration
MODEL_SERVER_HOST = "127.0.0.1"
MODEL_SERVER_PORT = 5001  # Different from kokoro to avoid conflicts

# Socket configuration
SOCKET_TIMEOUT = 120.0  # Increased for Maya1 generation time
SERVER_TIMEOUT = 120.0

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_MIN_NEW_TOKENS = 28  # At least 4 SNAC frames
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1

# Default voice description
# Note: Use detailed descriptions - short ones may be vocalized by the model
DEFAULT_DESCRIPTION = "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."

# Cache directories
CACHE_DIR = Path.home() / ".cache" / "maya"
LOG_DIR = CACHE_DIR / "logs"
LOG_FILE = LOG_DIR / "model_server.log"

# Thread pool configuration
NUM_WORKER_THREADS = 2  # Maya1 is heavy, limit concurrent requests

# Server startup configuration
SERVER_STARTUP_TIMEOUT = 60  # Seconds to wait for server to start (Maya1 is heavy)

# Logging
LOG_LEVEL = "INFO"
