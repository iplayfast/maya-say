#!/usr/bin/env python3
"""
Model server for Maya1 TTS.
Loads Maya1 and SNAC models into memory and handles synthesis requests.
"""

import socket
import threading
import logging
import sys
import traceback
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Any, Optional
import signal

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

from src.constants import (
    MODEL_NAME,
    SNAC_MODEL_NAME,
    MODEL_SERVER_HOST,
    MODEL_SERVER_PORT,
    NUM_WORKER_THREADS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MIN_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    CODE_END_TOKEN_ID,
    SAMPLE_RATE,
    WARMUP_SAMPLES,
    LOG_DIR,
)
from src.socket_protocol import SocketProtocol
from src.utils import build_prompt, extract_snac_codes, unpack_snac_from_7


# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'model_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ModelServer:
    """
    Central server managing Maya1 and SNAC models.
    Handles synthesis requests via TCP socket.
    """

    def __init__(self):
        self.maya_model = None
        self.tokenizer = None
        self.snac_model = None
        self.device = None
        self.model_lock = threading.Lock()
        self.running = False

        # Request queue and worker threads
        self.request_queue = Queue()
        self.num_workers = NUM_WORKER_THREADS
        self.workers = []

    def initialize_models(self):
        """Initialize Maya1 and SNAC models."""
        logger.info("Initializing models...")

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("Using CPU (this will be slow)")

        # Load Maya1 model
        logger.info(f"Loading Maya1 model from {MODEL_NAME}...")
        self.maya_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.maya_model.eval()
        logger.info(f"Maya1 model loaded successfully")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens in vocabulary")

        # Load SNAC decoder
        logger.info(f"Loading SNAC decoder from {SNAC_MODEL_NAME}...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to("cuda")
        logger.info("SNAC decoder loaded successfully")

        logger.info("All models initialized and ready")

    def synthesize(
        self,
        description: str,
        text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text using Maya1.

        Args:
            description: Voice description
            text: Text to synthesize
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            seed: Random seed for reproducible generation (optional)

        Returns:
            Audio samples as numpy array, or None on error
        """
        try:
            with self.model_lock:
                # Set random seed if provided
                if seed is not None:
                    logger.info(f"Setting seed: {seed}")
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    # Also set numpy seed for any numpy random operations
                    np.random.seed(seed)
                else:
                    logger.info("No seed specified - using random generation")

                # Build prompt
                prompt = build_prompt(self.tokenizer, description, text)
                logger.info(f"Synthesizing: description='{description[:50]}...', text='{text[:50]}...'")
                logger.info(f"Full prompt preview: {repr(prompt[:200])}")
                logger.debug(f"Prompt length: {len(prompt)} chars")

                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt")
                logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")

                # Move to device
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Generate
                logger.debug("Generating tokens...")
                with torch.inference_mode():
                    outputs = self.maya_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        do_sample=True,
                        eos_token_id=CODE_END_TOKEN_ID,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Extract generated tokens
                generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
                logger.debug(f"Generated {len(generated_ids)} tokens")

                # Debug: decode the generated text to see what the model produced
                try:
                    generated_text = self.tokenizer.decode(generated_ids[:50], skip_special_tokens=False)
                    logger.info(f"Generated text preview: {repr(generated_text[:200])}")
                except:
                    pass

                # Extract SNAC codes
                snac_tokens = extract_snac_codes(generated_ids)
                logger.debug(f"Extracted {len(snac_tokens)} SNAC tokens")

                if len(snac_tokens) < 7:
                    logger.error("Not enough SNAC tokens generated")
                    return None

                # Unpack SNAC tokens
                levels = unpack_snac_from_7(snac_tokens)
                frames = len(levels[0])
                logger.debug(f"Unpacked to {frames} frames: L1={len(levels[0])}, L2={len(levels[1])}, L3={len(levels[2])}")

                # Convert to tensors
                device = "cuda" if torch.cuda.is_available() else "cpu"
                codes_tensor = [
                    torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
                    for level in levels
                ]

                # Decode audio
                logger.debug("Decoding to audio...")
                with torch.inference_mode():
                    z_q = self.snac_model.quantizer.from_codes(codes_tensor)
                    audio = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

                # Trim warmup samples
                if len(audio) > WARMUP_SAMPLES:
                    audio = audio[WARMUP_SAMPLES:]

                duration_sec = len(audio) / SAMPLE_RATE
                logger.info(f"Synthesis complete: {len(audio)} samples ({duration_sec:.2f}s)")

                return audio

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            logger.error(traceback.format_exc())
            return None

    def synthesis_worker(self):
        """Worker thread for processing synthesis requests."""
        logger.info(f"Synthesis worker started: {threading.current_thread().name}")

        while self.running:
            try:
                # Get request from queue (with timeout to check running flag)
                try:
                    client_socket, request_data = self.request_queue.get(timeout=1.0)
                except Empty:
                    continue

                try:
                    # Extract parameters
                    description = request_data.get('description', '')
                    text = request_data.get('text', '')
                    max_new_tokens = request_data.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
                    min_new_tokens = request_data.get('min_new_tokens', DEFAULT_MIN_NEW_TOKENS)
                    temperature = request_data.get('temperature', DEFAULT_TEMPERATURE)
                    top_p = request_data.get('top_p', DEFAULT_TOP_P)
                    repetition_penalty = request_data.get('repetition_penalty', DEFAULT_REPETITION_PENALTY)
                    seed = request_data.get('seed', None)

                    # Synthesize
                    audio = self.synthesize(
                        description=description,
                        text=text,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        seed=seed,
                    )

                    if audio is not None:
                        # Send audio data
                        audio_bytes = audio.astype(np.float32).tobytes()
                        SocketProtocol.send_audio(client_socket, audio_bytes)
                        SocketProtocol.send_end(client_socket)
                    else:
                        SocketProtocol.send_error(client_socket, "Synthesis failed")

                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        SocketProtocol.send_error(client_socket, str(e))
                    except:
                        pass

                finally:
                    try:
                        client_socket.close()
                    except:
                        pass
                    self.request_queue.task_done()

            except Exception as e:
                logger.error(f"Worker error: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Synthesis worker stopped: {threading.current_thread().name}")

    def start_worker_threads(self):
        """Start worker threads for synthesis processing."""
        logger.info(f"Starting {self.num_workers} worker threads...")
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self.synthesis_worker,
                name=f"SynthesisWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        logger.info(f"{self.num_workers} worker threads started")

    def handle_client(self, client_socket: socket.socket, client_address):
        """Handle client connection."""
        logger.info(f"Client connected: {client_address}")

        try:
            # Receive request
            msg_type, request_data = SocketProtocol.receive_message(client_socket)

            if msg_type != 'json':
                logger.error(f"Expected JSON message, got {msg_type}")
                SocketProtocol.send_error(client_socket, "Expected JSON message")
                client_socket.close()
                return

            logger.debug(f"Received request: {request_data}")

            # Queue the request for processing
            self.request_queue.put((client_socket, request_data))

        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
            logger.error(traceback.format_exc())
            try:
                client_socket.close()
            except:
                pass

    def start(self):
        """Start the model server."""
        logger.info("Starting Maya1 Model Server...")

        # Initialize models
        self.initialize_models()

        # Start worker threads
        self.running = True
        self.start_worker_threads()

        # Create server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((MODEL_SERVER_HOST, MODEL_SERVER_PORT))
        server_socket.listen(5)

        logger.info(f"Server listening on {MODEL_SERVER_HOST}:{MODEL_SERVER_PORT}")
        logger.info("Ready to accept connections")

        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
            server_socket.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Accept connections
        try:
            while self.running:
                try:
                    client_socket, client_address = server_socket.accept()
                    # Handle each client in the main thread, but processing is done by workers
                    threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    ).start()
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
        finally:
            server_socket.close()
            logger.info("Server stopped")


def main():
    """Main entry point for model server."""
    server = ModelServer()
    server.start()


if __name__ == "__main__":
    main()
