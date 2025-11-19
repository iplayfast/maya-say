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

from src.tts_client import send_text, send_text_streaming, is_server_running, ensure_model_server
from src.presets import PresetManager
import numpy as np
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
        '--batch',
        type=str,
        default=None,
        help='Batch mode: path to file with one text per line'
    )

    parser.add_argument(
        '--batch-output-dir',
        type=str,
        default='batch_output',
        help='Directory for batch output files (default: batch_output)'
    )

    parser.add_argument(
        '-d', '--description',
        type=str,
        default=None,  # Will use preset or DEFAULT_DESCRIPTION
        help=f'Voice description (default: "{DEFAULT_DESCRIPTION}")'
    )

    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        help='Use a saved voice preset'
    )

    parser.add_argument(
        '--save-preset',
        type=str,
        default=None,
        help='Save current voice description as a preset'
    )

    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available voice presets'
    )

    parser.add_argument(
        '--delete-preset',
        type=str,
        default=None,
        help='Delete a saved preset'
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
        '--stream',
        action='store_true',
        help='Enable streaming mode (play audio as it generates)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (shows detailed chunk information)'
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
    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('src').setLevel(logging.DEBUG)

    # Initialize preset manager
    preset_manager = PresetManager()

    # Handle preset listing
    if args.list_presets:
        presets = preset_manager.list_presets()
        print("Available voice presets:")
        print()
        for name, description in sorted(presets.items()):
            print(f"  {name}:")
            print(f"    {description}")
            print()
        return 0

    # Handle preset deletion
    if args.delete_preset:
        if preset_manager.delete_preset(args.delete_preset):
            print(f"✓ Deleted preset '{args.delete_preset}'")
            return 0
        else:
            logger.error(f"Preset '{args.delete_preset}' not found")
            return 1

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

    # Require text or batch mode for synthesis
    if not args.text and not args.batch:
        parser.error("the following arguments are required: text or --batch (unless using --server-check or --kill)")

    # Resolve voice description (preset or explicit)
    if args.preset:
        description = preset_manager.get_preset(args.preset)
        if description is None:
            logger.error(f"Preset '{args.preset}' not found")
            logger.info("Use --list-presets to see available presets")
            return 1
        logger.info(f"Using preset '{args.preset}'")
    elif args.description:
        description = args.description
    else:
        description = DEFAULT_DESCRIPTION

    # Handle save preset
    if args.save_preset:
        preset_manager.save_preset(args.save_preset, description)
        print(f"✓ Saved preset '{args.save_preset}':")
        print(f"  {description}")
        # Continue with synthesis if text was provided, otherwise exit
        if not args.text and not args.batch:
            return 0

    # Ensure server is running (auto-start if needed)
    if not ensure_model_server():
        logger.error("Failed to start model server!")
        logger.error("\nYou can try starting it manually:")
        logger.error("  python -m src.model_server")
        return 1

    # Handle batch mode
    if args.batch:
        logger.info(f"Batch mode: processing file '{args.batch}'")

        # Read batch file
        try:
            with open(args.batch, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            logger.error(f"Batch file not found: {args.batch}")
            return 1
        except Exception as e:
            logger.error(f"Error reading batch file: {e}")
            return 1

        if not lines:
            logger.error("Batch file is empty")
            return 1

        logger.info(f"Processing {len(lines)} texts")

        # Create output directory
        output_dir = Path(args.batch_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Process each line
        success_count = 0
        for i, text in enumerate(lines, 1):
            logger.info(f"\n[{i}/{len(lines)}] Processing: '{text[:50]}...'")

            output_file = output_dir / f"batch_{i:03d}.wav"

            audio = send_text(
                text=text,
                description=description,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_tokens,
                seed=args.seed,
                stream=False,  # Batch mode doesn't use streaming
            )

            if audio is not None:
                save_audio(audio, str(output_file))
                success_count += 1
                logger.info(f"✓ Saved to {output_file}")
            else:
                logger.error(f"✗ Failed to synthesize item {i}")

        logger.info(f"\nBatch complete: {success_count}/{len(lines)} successful")
        return 0 if success_count == len(lines) else 1

    # Single text synthesis
    logger.info(f"Synthesizing: '{args.text[:50]}...'")
    logger.info(f"Voice description: '{description}'")
    if args.seed is not None:
        logger.info(f"Using seed: {args.seed}")
    if args.stream:
        logger.info("Streaming mode: ENABLED")
    if args.debug:
        logger.info("Debug mode: ENABLED - showing chunk details")
    if args.verbose or args.debug:
        logger.debug(f"Temperature: {args.temperature}, Top-p: {args.top_p}")

    if args.stream:
        # Streaming mode - play chunks as they arrive
        if args.output:
            # For file output, still collect all chunks
            logger.info("Streaming to file...")
            audio_chunks = []
            for chunk in send_text_streaming(
                text=args.text,
                description=description,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_tokens,
                seed=args.seed,
            ):
                audio_chunks.append(chunk)

            if not audio_chunks:
                logger.error("Synthesis failed - no chunks received")
                return 1

            audio = np.concatenate(audio_chunks)
            save_audio(audio, args.output)
        else:
            # Play chunks as they arrive
            logger.info("Starting streaming playback...")
            chunk_num = 0
            stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            stream.start()

            try:
                for chunk in send_text_streaming(
                    text=args.text,
                    description=description,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_tokens,
                    seed=args.seed,
                ):
                    chunk_num += 1
                    if args.debug:
                        logger.info(f"Playing chunk {chunk_num}: {len(chunk)} samples ({len(chunk)/SAMPLE_RATE:.2f}s)")
                    stream.write(chunk)

                # Wait for playback to finish
                stream.stop()
                stream.close()

                if chunk_num == 0:
                    logger.error("Synthesis failed - no chunks received")
                    return 1

            except Exception as e:
                logger.error(f"Streaming playback error: {e}")
                stream.stop()
                stream.close()
                return 1
    else:
        # Non-streaming mode
        audio = send_text(
            text=args.text,
            description=description,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_tokens,
            seed=args.seed,
            stream=False,
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
