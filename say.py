#!/usr/bin/env python3
"""
CLI interface for maya-say TTS.
"""

import argparse
import sys
import logging
from pathlib import Path

import soundfile as sf
import sounddevice as sd

from src.tts_client import send_text, is_server_running, ensure_model_server
from src.constants import (
    DEFAULT_DESCRIPTION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_MAX_NEW_TOKENS,
    SAMPLE_RATE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def play_audio(audio, sample_rate=SAMPLE_RATE):
    """Play audio using sounddevice."""
    try:
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"Error playing audio: {e}")


def save_audio(audio, output_file, sample_rate=SAMPLE_RATE):
    """Save audio to file."""
    try:
        sf.write(output_file, audio, sample_rate)
        logger.info(f"Saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving audio: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Maya1 Text-to-Speech CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Say text with default voice
  python say.py "Hello, this is Maya1 speaking!"

  # Use custom voice description
  python say.py "Hello!" -d "Young female voice, excited, high pitch"

  # Save to file instead of playing
  python say.py "Hello!" -o output.wav

  # Use emotion tags
  python say.py "This is funny! <laugh_harder> Really funny!"

  # Adjust generation parameters
  python say.py "Hello!" --temperature 0.6 --max-tokens 3000

Supported Emotion Tags:
  <laugh>          - Light laughter
  <laugh_harder>   - Intense laughter
  <giggle>         - Soft giggle
  <chuckle>        - Quiet chuckle
  <sigh>           - Sighing sound
  <gasp>           - Gasping sound
  <whisper>        - Whispered speech
  <cry>            - Crying emotion
  <angry>          - Angry tone
        """
    )

    parser.add_argument(
        'text',
        type=str,
        nargs='?',  # Make text optional
        help='Text to synthesize (can include emotion tags like <laugh>, <sigh>, etc.)'
    )

    parser.add_argument(
        '-d', '--description',
        type=str,
        default=DEFAULT_DESCRIPTION,
        help=f'Voice description (default: "{DEFAULT_DESCRIPTION}")'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: play audio)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Sampling temperature (default: {DEFAULT_TEMPERATURE})'
    )

    parser.add_argument(
        '--top-p',
        type=float,
        default=DEFAULT_TOP_P,
        help=f'Nucleus sampling parameter (default: {DEFAULT_TOP_P})'
    )

    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
        help=f'Repetition penalty (default: {DEFAULT_REPETITION_PENALTY})'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f'Maximum tokens to generate (default: {DEFAULT_MAX_NEW_TOKENS})'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible generation (default: None)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--server-check',
        action='store_true',
        help='Check if model server is running'
    )

    parser.add_argument(
        '--kill',
        action='store_true',
        help='Kill the running model server'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)

    # Kill server
    if args.kill:
        import subprocess
        import time
        try:
            # Find process on port 5001
            result = subprocess.run(['lsof', '-ti:5001'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                subprocess.run(['kill', pid])
                print(f"Killing model server (PID: {pid})...")

                # Wait for server to fully stop (up to 10 attempts, 0.5s each = 5s max)
                for i in range(10):
                    time.sleep(0.5)
                    if not is_server_running():
                        print(f"✓ Server stopped after {(i+1)*0.5:.1f}s")
                        return 0

                # Check one more time after full wait
                if is_server_running():
                    print("⚠ Server may still be shutting down...")
                    return 1
                else:
                    print("✓ Server stopped")
                    return 0
            else:
                print("✗ No model server found running")
                return 1
        except Exception as e:
            logger.error(f"Error killing server: {e}")
            return 1

    # Check server status
    if args.server_check:
        if is_server_running():
            print("✓ Model server is running")
            return 0
        else:
            print("✗ Model server is not running")
            print(f"\nStart the server with:")
            print(f"  python -m src.model_server")
            return 1

    # Require text for synthesis
    if not args.text:
        parser.error("the following arguments are required: text (unless using --server-check or --kill)")

    # Ensure server is running (auto-start if needed)
    if not ensure_model_server():
        logger.error("Failed to start model server!")
        logger.error("\nYou can try starting it manually:")
        logger.error("  python -m src.model_server")
        return 1

    # Synthesize
    logger.info(f"Synthesizing: '{args.text[:50]}...'")
    logger.info(f"Voice description: '{args.description}'")
    if args.seed is not None:
        logger.info(f"Using seed: {args.seed}")
    if args.verbose:
        logger.debug(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    audio = send_text(
        text=args.text,
        description=args.description,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_tokens,
        seed=args.seed,
    )

    if audio is None:
        logger.error("Synthesis failed")
        return 1

    # Output
    if args.output:
        save_audio(audio, args.output)
    else:
        logger.info("Playing audio...")
        play_audio(audio)

    return 0


if __name__ == '__main__':
    sys.exit(main())
