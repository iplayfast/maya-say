#!/usr/bin/env python3
"""
TTS client library for Maya1.
Handles communication with the model server.
"""

import socket
import subprocess
import sys
import os
import time
import tempfile
import numpy as np
from typing import Optional
import logging

from src.constants import (
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    SOCKET_TIMEOUT,
    DEFAULT_DESCRIPTION,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    SERVER_STARTUP_TIMEOUT,
    LOG_FILE,
)
from src.socket_protocol import SocketProtocol

logger = logging.getLogger(__name__)


def send_text_streaming(
    text: str,
    description: str = DEFAULT_DESCRIPTION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    seed: Optional[int] = None,
    timeout: float = SOCKET_TIMEOUT,
):
    """
    Send text for synthesis and yield audio chunks as they arrive.

    Yields:
        Audio chunk as numpy array
    """
    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))

        # Send request
        request = {
            'text': text,
            'description': description,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'seed': seed,
            'stream': True,  # Always use streaming
        }
        SocketProtocol.send_json(sock, request)

        # Receive and yield chunks as they arrive
        chunk_num = 0
        for msg_type, data in SocketProtocol.receive_stream(sock):
            if msg_type == 'audio':
                chunk_num += 1

                # Convert to audio array
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Debug: Show chunk info
                chunk_samples = len(audio_chunk)
                chunk_duration = chunk_samples / 24000
                logger.debug(f"Received chunk {chunk_num}: {chunk_samples} samples ({chunk_duration:.2f}s)")

                yield audio_chunk
            elif msg_type == 'error':
                logger.error(f"Server error: {data}")
                return
            elif msg_type == 'end':
                break

        sock.close()
        logger.info(f"Received {chunk_num} audio chunks total")

    except socket.timeout:
        logger.error("Request timed out")
        return
    except ConnectionRefusedError:
        logger.error(f"Could not connect to server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        logger.error("Make sure the model server is running")
        return
    except Exception as e:
        logger.error(f"Client error: {e}")
        return


def send_text(
    text: str,
    description: str = DEFAULT_DESCRIPTION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    seed: Optional[int] = None,
    stream: bool = False,
    timeout: float = SOCKET_TIMEOUT,
) -> Optional[np.ndarray]:
    """
    Send text to model server for synthesis.

    Args:
        text: Text to synthesize
        description: Voice description
        max_new_tokens: Maximum tokens to generate
        min_new_tokens: Minimum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        repetition_penalty: Penalty for repetition
        seed: Random seed for reproducible generation (optional)
        stream: Enable streaming mode (receive audio chunks as generated)
        timeout: Socket timeout in seconds

    Returns:
        Audio samples as numpy array, or None on error
    """
    try:
        # Connect to server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((MODEL_SERVER_HOST, MODEL_SERVER_PORT))

        # Send request
        request = {
            'text': text,
            'description': description,
            'max_new_tokens': max_new_tokens,
            'min_new_tokens': min_new_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'seed': seed,
            'stream': stream,
        }
        SocketProtocol.send_json(sock, request)

        # Receive response
        audio_chunks = []
        chunk_num = 0
        for msg_type, data in SocketProtocol.receive_stream(sock):
            if msg_type == 'audio':
                chunk_num += 1
                audio_chunks.append(data)

                # Debug: Show chunk info
                chunk_samples = len(data) // 4  # 4 bytes per float32
                chunk_duration = chunk_samples / 24000  # 24kHz sample rate
                logger.debug(f"Received chunk {chunk_num}: {chunk_samples} samples ({chunk_duration:.2f}s)")
            elif msg_type == 'error':
                logger.error(f"Server error: {data}")
                return None
            elif msg_type == 'end':
                break

        sock.close()

        if not audio_chunks:
            logger.error("No audio data received")
            return None

        # Combine audio chunks
        logger.info(f"Received {chunk_num} audio chunks total")
        audio_bytes = b''.join(audio_chunks)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        return audio

    except socket.timeout:
        logger.error("Request timed out")
        return None
    except ConnectionRefusedError:
        logger.error(f"Could not connect to server at {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        logger.error("Make sure the model server is running")
        return None
    except Exception as e:
        logger.error(f"Client error: {e}")
        return None


def is_server_running() -> bool:
    """
    Check if the model server is running.

    Returns:
        True if server is running, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        result = sock.connect_ex((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        sock.close()
        return result == 0
    except:
        return False


def ensure_model_server() -> bool:
    """
    Ensure model server is running, start it if needed.

    Returns:
        True if server is running, False if startup failed
    """
    try:
        # Check if server is already running
        if is_server_running():
            logger.info("Model server already running")
            return True

        # Start model server
        logger.info("Model server not running, starting it...")
        logger.info("This may take a minute on first run (downloading models)...")

        # Get the absolute path to the current project directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create log directory
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Create a script to launch the server
        launch_script = f"""#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('{LOG_FILE}'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("launcher")
logger.info("==== MODEL SERVER LAUNCHER STARTED ====")

# Add project directory to the Python path
project_dir = {repr(current_dir)}
logger.info(f"Project directory: {{project_dir}}")
sys.path.insert(0, project_dir)

try:
    logger.info("Starting model server")
    from src.model_server import ModelServer

    server = ModelServer()
    server.start()
except Exception as e:
    logger.error(f"Server startup failed: {{e}}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)
"""

        # Write the script to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(launch_script)
            temp_script = f.name

        # Make the script executable
        os.chmod(temp_script, 0o755)

        # Start the server in a separate process
        process = subprocess.Popen(
            [sys.executable, temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Detach the process
        )

        # Wait for server to become available
        logger.info("Waiting for model server to start...")
        start_time = time.time()
        attempts = 0

        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            if is_server_running():
                logger.info("Model server started successfully")
                os.unlink(temp_script)  # Clean up temp file
                return True

            # Check if process is still running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server process exited with code {process.returncode}")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                logger.error(f"Check {LOG_FILE} for more details")
                os.unlink(temp_script)  # Clean up temp file
                return False

            # Exponential backoff
            sleep_time = min(2 ** attempts * 0.5, 2.0)
            time.sleep(sleep_time)
            attempts += 1

        # If we get here, timeout occurred
        logger.error(f"Timeout waiting for model server to start (waited {SERVER_STARTUP_TIMEOUT}s)")
        logger.error(f"Check {LOG_FILE} for error details")
        os.unlink(temp_script)  # Clean up temp file
        return False

    except Exception as e:
        logger.error(f"Failed to ensure model server: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
