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
from typing import Dict, Any, Optional, Iterator
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
    USE_LMDEPLOY,
    LMDEPLOY_MEMORY_UTIL,
    LMDEPLOY_TP,
    LMDEPLOY_ENABLE_PREFIX_CACHING,
    LMDEPLOY_QUANT_POLICY,
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

# Try to import LMdeploy if enabled
LMDEPLOY_AVAILABLE = False
if USE_LMDEPLOY:
    try:
        from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
        LMDEPLOY_AVAILABLE = True
        logger.info("LMdeploy is available and will be used")
    except ImportError:
        LMDEPLOY_AVAILABLE = False
        logger.warning("LMdeploy not installed. Install with: pip install lmdeploy")
        logger.warning("Falling back to transformers backend")


class ModelServer:
    """
    Central server managing Maya1 and SNAC models.
    Handles synthesis requests via TCP socket.
    """

    def __init__(self):
        self.maya_model = None
        self.lmdeploy_pipe = None
        self.tokenizer = None
        self.snac_model = None
        self.device = None
        self.model_lock = threading.Lock()
        self.running = False
        self.use_lmdeploy = USE_LMDEPLOY and LMDEPLOY_AVAILABLE

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

        if self.use_lmdeploy:
            # Load Maya1 with LMdeploy
            logger.info(f"Loading Maya1 model with LMdeploy from {MODEL_NAME}...")
            logger.info(f"LMdeploy config: memory_util={LMDEPLOY_MEMORY_UTIL}, tp={LMDEPLOY_TP}, "
                       f"prefix_caching={LMDEPLOY_ENABLE_PREFIX_CACHING}, quant_policy={LMDEPLOY_QUANT_POLICY}")

            backend_config = TurbomindEngineConfig(
                cache_max_entry_count=LMDEPLOY_MEMORY_UTIL,  # This is supposed to be memory utilization
                session_len=4096,  # Maximum session length
                tp=LMDEPLOY_TP,
                enable_prefix_caching=LMDEPLOY_ENABLE_PREFIX_CACHING,
                quant_policy=LMDEPLOY_QUANT_POLICY
            )
            self.lmdeploy_pipe = pipeline(MODEL_NAME, backend_config=backend_config)
            logger.info("Maya1 model loaded successfully with LMdeploy")

            # Load tokenizer for prompt building
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens in vocabulary")
        else:
            # Load Maya1 with transformers
            logger.info(f"Loading Maya1 model with transformers from {MODEL_NAME}...")
            self.maya_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.maya_model.eval()
            logger.info("Maya1 model loaded successfully with transformers")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens in vocabulary")

        # Load SNAC decoder (same for both backends)
        logger.info(f"Loading SNAC decoder from {SNAC_MODEL_NAME}...")
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval()
        if torch.cuda.is_available():
            self.snac_model = self.snac_model.to("cuda")
        logger.info("SNAC decoder loaded successfully")

        backend_name = "LMdeploy" if self.use_lmdeploy else "transformers"
        logger.info(f"All models initialized and ready (backend: {backend_name})")

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

                if self.use_lmdeploy:
                    # Generate with LMdeploy
                    logger.debug("Generating tokens with LMdeploy...")
                    gen_config = GenerationConfig(
                        top_p=top_p,
                        top_k=40,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        stop_token_ids=[CODE_END_TOKEN_ID],
                        do_sample=True,
                        min_p=0.0
                    )

                    responses = self.lmdeploy_pipe([prompt], gen_config=gen_config, do_preprocess=False)
                    generated_ids = responses[0].token_ids
                    logger.debug(f"Generated {len(generated_ids)} tokens")
                else:
                    # Generate with transformers
                    # Tokenize
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    logger.debug(f"Input tokens: {inputs['input_ids'].shape[1]}")

                    # Move to device
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    # Generate
                    logger.debug("Generating tokens with transformers...")
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

                    # Extract generated tokens (skip input prompt)
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

    def synthesize_streaming(
        self,
        description: str,
        text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        seed: Optional[int] = None,
        chunk_frames: int = 50,  # Number of SNAC frames per audio chunk
    ) -> Iterator[np.ndarray]:
        """
        Synthesize speech with streaming output.
        Yields audio chunks as they're generated.

        Args:
            description: Voice description
            text: Text to synthesize
            max_new_tokens: Maximum tokens to generate
            min_new_tokens: Minimum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repetition
            seed: Random seed for reproducible generation (optional)
            chunk_frames: Number of SNAC frames to accumulate before yielding audio

        Yields:
            Audio chunk as numpy array
        """
        try:
            with self.model_lock:
                # Set random seed if provided
                if seed is not None:
                    logger.info(f"Setting seed: {seed}")
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    np.random.seed(seed)
                else:
                    logger.info("No seed specified - using random generation")

                # Build prompt
                prompt = build_prompt(self.tokenizer, description, text)
                logger.info(f"Synthesizing (streaming): description='{description[:50]}...', text='{text[:50]}...'")
                logger.debug(f"Chunk size: {chunk_frames} frames")

                # Generate tokens (non-streaming for now - token generation is fast)
                if self.use_lmdeploy:
                    logger.debug("Generating tokens with LMdeploy...")
                    gen_config = GenerationConfig(
                        top_p=top_p,
                        top_k=40,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        stop_token_ids=[CODE_END_TOKEN_ID],
                        do_sample=True,
                        min_p=0.0
                    )
                    responses = self.lmdeploy_pipe([prompt], gen_config=gen_config, do_preprocess=False)
                    generated_ids = responses[0].token_ids
                else:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    logger.debug("Generating tokens with transformers...")
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
                    generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()

                logger.debug(f"Generated {len(generated_ids)} tokens")

                # Extract SNAC codes
                snac_tokens = extract_snac_codes(generated_ids)
                logger.debug(f"Extracted {len(snac_tokens)} SNAC tokens")

                if len(snac_tokens) < 7:
                    logger.error("Not enough SNAC tokens generated")
                    return

                # Stream audio in chunks
                total_frames = len(snac_tokens) // 7
                logger.info(f"Streaming {total_frames} frames in chunks of {chunk_frames}")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                first_chunk = True

                for start_frame in range(0, total_frames, chunk_frames):
                    end_frame = min(start_frame + chunk_frames, total_frames)

                    # Extract chunk of SNAC tokens
                    chunk_tokens = snac_tokens[start_frame * 7:end_frame * 7]

                    # Unpack this chunk
                    levels = unpack_snac_from_7(chunk_tokens)

                    # Convert to tensors
                    codes_tensor = [
                        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
                        for level in levels
                    ]

                    # Decode audio chunk
                    with torch.inference_mode():
                        z_q = self.snac_model.quantizer.from_codes(codes_tensor)
                        audio_chunk = self.snac_model.decoder(z_q)[0, 0].cpu().numpy()

                    # Trim warmup samples from first chunk only
                    if first_chunk and len(audio_chunk) > WARMUP_SAMPLES:
                        audio_chunk = audio_chunk[WARMUP_SAMPLES:]
                        first_chunk = False

                    logger.debug(f"Yielding audio chunk: {len(audio_chunk)} samples ({len(audio_chunk)/SAMPLE_RATE:.2f}s)")
                    yield audio_chunk

                logger.info("Streaming synthesis complete")

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            logger.error(traceback.format_exc())
            return

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
                    stream = request_data.get('stream', False)  # Streaming mode flag

                    if stream:
                        # Streaming mode - send audio chunks as they're generated
                        chunk_count = 0
                        for audio_chunk in self.synthesize_streaming(
                            description=description,
                            text=text,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            seed=seed,
                        ):
                            audio_bytes = audio_chunk.astype(np.float32).tobytes()
                            SocketProtocol.send_audio(client_socket, audio_bytes)
                            chunk_count += 1

                        logger.debug(f"Sent {chunk_count} audio chunks")
                        SocketProtocol.send_end(client_socket)
                    else:
                        # Non-streaming mode - send all audio at once
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
