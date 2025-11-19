# Voice Description Examples

This document provides tested voice description examples for Maya1 TTS. Use these as templates or inspiration for creating your own voices.

## Table of Contents
- [General Guidelines](#general-guidelines)
- [Male Voices](#male-voices)
- [Female Voices](#female-voices)
- [Character Voices](#character-voices)
- [Professional Voices](#professional-voices)
- [Emotional Variations](#emotional-variations)
- [Age Variations](#age-variations)
- [Tips and Best Practices](#tips-and-best-practices)

## General Guidelines

**✓ DO:**
- Use detailed, multi-sentence descriptions
- Always end descriptions with a period (reduces vocalization)
- Include: age, gender, accent, pitch, timbre, pacing
- Be specific about tone and emotion

**✗ DON'T:**
- Use short descriptions like "male voice" (may be vocalized)
- Forget the period at the end
- Use contradictory attributes

**Formula:**
```
[Gender] [Age] voice, [Accent], [Pitch/Timbre], [Pacing], [Tone/Emotion].
```

## Male Voices

### Young Adult Male
```
Young male voice in his twenties, american accent, energetic tone, moderate pitch, clear timbre, natural pacing.
```

### Professional Male (30s)
```
Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.
```

### Middle-Aged Male
```
Middle-aged male voice, canadian accent, warm and friendly tone, moderate pitch, calm pacing.
```

### Mature Male
```
Elderly male voice in his sixties, british accent, distinguished tone, lower pitch, slow and deliberate pacing.
```

### Deep Male
```
Deep male voice, commanding presence, low pitch, resonant timbre, slow authoritative pacing.
```

### Casual Male
```
Relaxed male voice in his thirties, american accent, casual friendly tone, normal pitch, easygoing pacing.
```

### Formal Male
```
Professional male voice, british accent, formal articulation, moderate pitch, precise pacing.
```

## Female Voices

### Young Female
```
Young female voice in her twenties, american accent, bright tone, higher pitch, energetic pacing.
```

### Professional Female
```
Clear female voice in her thirties, american accent, professional tone, moderate pitch, confident pacing.
```

### Mature Female
```
Middle-aged female voice, warm and experienced tone, moderate pitch, measured pacing.
```

### Energetic Female
```
Young female voice, excited tone, high pitch, energetic pacing, enthusiastic delivery.
```

### Calm Female
```
Soothing female voice, calm and gentle tone, soft timbre, slow relaxed pacing.
```

### Sophisticated Female
```
Elegant female voice, british accent, sophisticated tone, clear articulation, refined pacing.
```

## Character Voices

### News Anchor
```
Professional news anchor voice, clear and authoritative, neutral accent, moderate pitch, steady informative pacing.
```

### Storyteller
```
Engaging storyteller voice, expressive tone, varied pacing, warm timbre, captivating delivery.
```

### Teacher
```
Patient teacher voice, clear and encouraging tone, moderate pitch, steady instructional pacing.
```

### Radio Host
```
Charismatic radio host voice, friendly and energetic, clear delivery, engaging tone, dynamic pacing.
```

### Audiobook Narrator
```
Professional narrator voice, clear articulation, neutral accent, moderate pacing, consistent delivery.
```

### Documentary Narrator
```
Authoritative documentary narrator, deep resonant voice, measured pacing, serious informative tone.
```

### Sports Announcer
```
Energetic sports announcer voice, excited tone, fast dynamic pacing, enthusiastic delivery.
```

## Professional Voices

### Customer Service
```
Friendly customer service voice, warm and helpful tone, clear articulation, patient pacing.
```

### Technical Support
```
Professional technical voice, calm and reassuring tone, clear explanations, moderate pacing.
```

### Corporate Presenter
```
Professional corporate voice, confident and polished tone, clear delivery, authoritative pacing.
```

### Voice Assistant
```
Friendly AI assistant voice, helpful and clear tone, neutral accent, consistent pacing.
```

## Emotional Variations

### Happy/Upbeat
```
Cheerful voice, happy and upbeat tone, bright timbre, lively pacing, positive energy.
```

### Calm/Relaxed
```
Calm and relaxed voice, soothing tone, gentle timbre, slow peaceful pacing.
```

### Serious/Formal
```
Serious professional voice, formal tone, clear articulation, measured deliberate pacing.
```

### Enthusiastic
```
Enthusiastic voice, excited and energetic tone, fast pacing, dynamic delivery.
```

### Mysterious
```
Mysterious voice, intriguing tone, moderate pitch, deliberate pacing, enigmatic quality.
```

## Age Variations

### Child/Young Teen (Note: Maya1 may not accurately represent children)
```
Young teenager voice, youthful tone, higher pitch, energetic pacing, casual delivery.
```

### Young Adult (20s-30s)
```
Young adult voice in the twenties, vibrant tone, moderate pitch, natural pacing.
```

### Middle-Aged (40s-50s)
```
Middle-aged voice in the forties, experienced tone, warm timbre, confident pacing.
```

### Senior (60s+)
```
Elderly voice in the sixties, wise and measured tone, slower pacing, distinguished quality.
```

## Tips and Best Practices

### Combining Attributes
You can combine multiple characteristics for unique voices:
```
Young female voice with slight raspy quality, american accent, confident tone, moderate pitch, conversational pacing.
```

### Using Emotion Tags
Descriptions work well with emotion tags in the text:
```
Description: "Professional narrator voice, clear articulation, neutral accent."
Text: "This is incredible! <laugh> Absolutely amazing! <gasp> I can't believe it!"
```

### Consistency Across Sessions
Use `--seed` with the same description for reproducible voices:
```bash
python say.py "Text 1" -d "Your description" --seed 42
python say.py "Text 2" -d "Your description" --seed 42  # Same voice
```

### Save Your Favorites
When you find a voice you like, save it as a preset:
```bash
python say.py --save-preset my_favorite -d "Your perfect description here."
```

### Testing Variations
Create multiple variations to find what works:
```bash
# Test different pacing
python say.py "Test" -d "Male voice, fast pacing." -o fast.wav
python say.py "Test" -d "Male voice, slow pacing." -o slow.wav
python say.py "Test" -d "Male voice, moderate pacing." -o moderate.wav
```

### Common Issues

**Description is spoken as text:**
- Make sure description ends with a period
- Use longer, more detailed descriptions
- Avoid very short descriptions

**Voice sounds inconsistent:**
- Use `--seed` for reproducibility
- Keep descriptions detailed and specific
- Avoid contradictory attributes

**Voice doesn't match description:**
- Maya1 is non-deterministic without seed
- Try adding more detail to the description
- Experiment with different phrasings

## Quick Reference

### Essential Components
1. **Gender & Age**: "Male voice in his thirties"
2. **Accent**: "american/british/canadian accent"
3. **Pitch**: "low/moderate/high pitch"
4. **Timbre**: "warm/clear/resonant timbre"
5. **Pacing**: "slow/moderate/fast pacing"
6. **Tone**: "professional/casual/friendly tone"
7. **Period**: Always end with "."

### Template
```
[Gender] [age] voice, [accent], [tone], [pitch], [timbre], [pacing].
```

### Example
```
Middle-aged female voice, american accent, professional tone, moderate pitch, warm timbre, natural pacing.
```

## Contributing

Found a great voice description? Consider sharing it! You can:
1. Save it as a preset for personal use
2. Add it to this document via pull request
3. Share it with the community

## Additional Resources

- Maya1 Model Card: https://huggingface.co/maya-research/maya1
- Emotion Tags Reference: See README.md
- Voice Presets: `python say.py --list-presets`
