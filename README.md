# maya-say

A Text-to-Speech system based on Maya1, implementing a persistent model server architecture similar to kokoro-say.

## Architecture

maya-say uses a **two-part architecture**:

1. **Model Server** (Part 1): Loads Maya1 and SNAC models into memory once and keeps them resident
2. **Client Interface** (Part 2): Lightweight clients send synthesis requests to the server

This architecture provides:
- Fast synthesis after initial model load
- Efficient resource usage (models loaded once)
- Concurrent request handling
- Simple client interface

## Features

- Natural language voice descriptions (no voice files needed!)
- Emotion tags support (`<laugh>`, `<sigh>`, `<whisper>`, etc.)
- Streaming audio generation
- Save to file or play directly
- Adjustable generation parameters
- Persistent model server for fast synthesis

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (16GB+ VRAM)
- CPU mode supported but slow

## Installation

1. Clone or navigate to the repository:
   ```bash
   cd maya-say
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. The models will be downloaded automatically on first use:
   - Maya1 model: ~6GB
   - SNAC decoder: ~100MB

## Usage

### Quick Start

The model server will start automatically when needed:

```bash
# Just run this - the server starts automatically!
python say.py "Hello! This is Maya1 speaking."
```

**Note:** The first run will take a few minutes to download models (~6GB). Subsequent runs are much faster.

### Basic Usage

```bash
# Basic usage with default voice
python say.py "Hello! This is Maya1 speaking."

# Custom voice description
python say.py "Hello!" -d "Young female voice, excited, high pitch"

# With emotion tags
python say.py "This is funny! <laugh_harder> Really funny!"

# Save to file
python say.py "Hello world!" -o output.wav

# Adjust generation parameters
python say.py "Hello!" --temperature 0.6 --max-tokens 3000

# Enable streaming mode (lower latency)
python say.py "Long text here..." --stream

# Use a seed for reproducible output
python say.py "Hello!" --seed 42

# Batch processing from a file
python say.py --batch texts.txt -d "Friendly voice"

# Verbose output
python say.py "Hello!" -v
```

### Batch Processing

Process multiple texts efficiently:

```bash
# Create a text file with one text per line
cat > batch.txt << EOF
Hello, this is the first sentence.
Welcome to batch processing!
This is much more efficient than individual requests.
EOF

# Process all texts
python say.py --batch batch.txt -d "Clear female voice"

# Custom output directory
python say.py --batch batch.txt --batch-output-dir my_outputs

# Use same seed for all items (reproducible)
python say.py --batch batch.txt --seed 42
```

**Features:**
- Processes multiple texts sequentially
- Outputs to `batch_output/batch_001.wav`, `batch_002.wav`, etc.
- Uses same voice description for all items
- Efficient for generating multiple audio files

### Voice Presets

Save and reuse your favorite voice descriptions:

```bash
# List available presets
python say.py --list-presets

# Use a preset
python say.py "Hello!" --preset female_energetic

# Save current description as a preset
python say.py --save-preset my_voice -d "Custom voice description here"

# Use your saved preset
python say.py "Hello!" --preset my_voice

# Delete a preset
python say.py --delete-preset my_voice
```

**Built-in presets:**
- `male_american` - Realistic male, 30s, American accent
- `female_american` - Young female, clear, American accent
- `male_british` - Middle-aged male, British accent, formal
- `female_british` - Young female, British accent, professional
- `male_deep` - Deep commanding voice
- `female_energetic` - Excited, energetic female voice
- `male_calm` - Calm, soothing male voice
- `narrator` - Professional narrator voice

Presets are stored in `~/.config/maya/presets.json`

### Manual Server Control

While the server starts automatically, you can also:

```bash
# Check server status
python say.py --server-check

# Start server manually (optional)
python -m src.model_server
```

### Testing

Test scripts are provided to verify functionality:

```bash
# Test the CLI with various features
./test_say.sh

# Test the standalone quickstart example
./test_quickstart.sh
```

## Streaming Mode

maya-say supports streaming audio generation for lower latency:

```bash
# Enable streaming for faster perceived response
python say.py "Your text here" --stream
```

**How it works:**
- **Non-streaming (default)**: Generate all tokens → Decode all audio → Send/play
- **Streaming**: Generate all tokens → Decode in chunks → Send/play as ready

**Benefits:**
- Lower perceived latency (audio starts playing sooner)
- More efficient memory usage
- Better for long texts and interactive applications

**Note:** Total synthesis time is the same, but streaming reduces time-to-first-audio.

## Voice Descriptions

Maya1 uses natural language to describe voices. **Important:** Use detailed descriptions - short ones may be vocalized by the model.

**Good examples (detailed):**
- `"Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."`
- `"Young female voice, excited tone, high pitch, energetic pacing"`
- `"Middle-aged male voice, warm timbre, low pitch, conversational pacing"`
- `"Elderly male voice, british accent, slow and formal delivery"`

**Avoid (too short):**
- `"Young female voice"` ← May be spoken as text
- `"Male voice"` ← Too minimal

**For many more examples, see [VOICE_EXAMPLES.md](VOICE_EXAMPLES.md)**

**Demo different voices:**
```bash
./demo_voices.sh  # Generates 8 different voice samples
```

## Emotion Tags

Supported emotion tags include:
- `<laugh>`, `<laugh_harder>`
- `<giggle>`, `<chuckle>`
- `<sigh>`
- `<gasp>`
- `<whisper>`
- `<cry>`
- `<angry>`

## Command-Line Options

```
usage: say.py [-h] [-d DESCRIPTION] [-o OUTPUT] [--temperature TEMPERATURE]
              [--top-p TOP_P] [--repetition-penalty REPETITION_PENALTY]
              [--max-tokens MAX_TOKENS] [-v] [--server-check]
              text

positional arguments:
  text                  Text to synthesize

optional arguments:
  -h, --help            Show this help message
  -d DESCRIPTION        Voice description (default: male, 30s, american)
  -o OUTPUT             Output file path (default: play audio)
  --temperature         Sampling temperature (default: 0.4)
  --top-p               Nucleus sampling parameter (default: 0.9)
  --repetition-penalty  Repetition penalty (default: 1.1)
  --max-tokens          Maximum tokens to generate (default: 2048)
  -v, --verbose         Enable verbose logging
  --server-check        Check if model server is running
```

## Architecture Details

### File Structure

```
maya-say/
├── say.py                  # CLI entry point
├── quickstart.py           # Original standalone example
├── requirements.txt        # Dependencies
├── README.md              # This file
└── src/
    ├── __init__.py        # Package marker
    ├── model_server.py    # Main model management server
    ├── tts_client.py      # Client library
    ├── constants.py       # Configuration
    ├── socket_protocol.py # IPC protocol
    └── utils.py           # Helper functions
```

### Communication Flow

1. Client sends JSON request to model server (TCP socket)
2. Model server queues request for worker thread
3. Worker thread:
   - Builds prompt from description and text
   - Generates tokens with Maya1
   - Extracts SNAC codes
   - Decodes with SNAC
   - Streams audio back to client
4. Client receives audio and plays/saves

### Logs

Server logs are stored in `~/.cache/maya/logs/model_server.log`

## Performance Optimization (Optional)

For significantly faster inference (4-10x speedup), you can use **LMdeploy** instead of the standard transformers backend:

### Using LMdeploy

LMdeploy provides optimized inference for large language models:

- **4x faster** than transformers without batching
- **10x+ faster** with batching enabled
- Lower memory usage
- Better throughput for concurrent requests
- KV-cache quantization for reduced VRAM usage

**Installation:**
```bash
pip install lmdeploy
```

**Enabling LMdeploy:**

1. Install LMdeploy (see above)
2. Edit `src/constants.py` and set:
   ```python
   USE_LMDEPLOY = True
   ```
3. Optional: Adjust configuration parameters:
   ```python
   LMDEPLOY_MEMORY_UTIL = 0.9  # GPU memory utilization (0.0-1.0)
   LMDEPLOY_TP = 1              # Tensor parallelism (for multi-GPU)
   LMDEPLOY_ENABLE_PREFIX_CACHING = True  # Cache common prompt prefixes
   LMDEPLOY_QUANT_POLICY = 0    # KV-cache quantization: 0 (disabled), 4, 8
   ```
4. Restart the server:
   ```bash
   python say.py --kill
   python say.py "Test with LMdeploy!"
   ```

**Configuration Options:**
- `LMDEPLOY_MEMORY_UTIL`: Controls GPU VRAM usage (e.g., 0.9 = 90%)
- `LMDEPLOY_TP`: Tensor parallelism for multi-GPU setups (default: 1)
- `LMDEPLOY_ENABLE_PREFIX_CACHING`: Caches common prompt parts for faster repeated generations
- `LMDEPLOY_QUANT_POLICY`: KV-cache quantization (0=disabled, 4=4-bit, 8=8-bit)

**Trade-offs:**
- Additional dependency (~2GB install)
- Slightly different generation behavior (may need parameter tuning)
- No `min_new_tokens` support (uses different stopping strategy)

**Performance Notes:**
Based on [FastMaya benchmarks](https://github.com/ysharma3501/FastMaya):
- RTF (Real-Time Factor) of 0.02 on A100 (50x faster than realtime)
- Can generate 50 seconds of audio in 1 second with batching
- Roughly 2x faster than vLLM implementation

## Differences from kokoro-say

- **No voice files**: Maya1 uses natural language descriptions instead of .pt files
- **Simpler architecture**: No need for per-voice servers
- **Heavier model**: Maya1 is 3B parameters vs Kokoro's 82M
- **More flexible**: Voice characteristics controlled by description text
- **Emotion support**: Built-in emotion tags

## License

Apache 2.0 (following Maya1's license)

## Credits

Based on:
- [Maya1](https://huggingface.co/maya-research/maya1) by Maya Research
- [SNAC](https://github.com/hubertsiuzdak/snac) by Hubert Siuzdak
- Architecture inspired by [kokoro-say](https://github.com/iplayfast/kokoro-say)
