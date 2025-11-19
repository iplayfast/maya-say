#!/bin/bash
# Demonstration script for various voice examples

set -e

CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
PURPLE='\033[1;35m'
NC='\033[0m'

echo -e "${CYAN}=========================================${NC}"
echo -e "${CYAN}   Voice Description Demo               ${NC}"
echo -e "${CYAN}=========================================${NC}\n"

TEST_TEXT="Hello! This is a demonstration of different voice styles in Maya1."

# Create output directory
mkdir -p voice_demos
echo -e "${YELLOW}Outputs will be saved to: voice_demos/${NC}\n"

# Demo 1: Professional Male
echo -e "${PURPLE}[1/8] Professional Male (30s)${NC}"
DESC="Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/01_professional_male.wav
echo -e "${GREEN}✓ Saved to voice_demos/01_professional_male.wav${NC}\n"

# Demo 2: Energetic Female
echo -e "${PURPLE}[2/8] Energetic Female${NC}"
DESC="Young female voice, excited tone, high pitch, energetic pacing, enthusiastic delivery."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/02_energetic_female.wav
echo -e "${GREEN}✓ Saved to voice_demos/02_energetic_female.wav${NC}\n"

# Demo 3: Deep Male
echo -e "${PURPLE}[3/8] Deep Male${NC}"
DESC="Deep male voice, commanding presence, low pitch, resonant timbre, slow authoritative pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/03_deep_male.wav
echo -e "${GREEN}✓ Saved to voice_demos/03_deep_male.wav${NC}\n"

# Demo 4: British Female
echo -e "${PURPLE}[4/8] British Female${NC}"
DESC="Elegant female voice, british accent, sophisticated tone, clear articulation, refined pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/04_british_female.wav
echo -e "${GREEN}✓ Saved to voice_demos/04_british_female.wav${NC}\n"

# Demo 5: News Anchor
echo -e "${PURPLE}[5/8] News Anchor${NC}"
DESC="Professional news anchor voice, clear and authoritative, neutral accent, moderate pitch, steady informative pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/05_news_anchor.wav
echo -e "${GREEN}✓ Saved to voice_demos/05_news_anchor.wav${NC}\n"

# Demo 6: Calm Voice
echo -e "${PURPLE}[6/8] Calm Soothing${NC}"
DESC="Calm and relaxed voice, soothing tone, gentle timbre, slow peaceful pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/06_calm_soothing.wav
echo -e "${GREEN}✓ Saved to voice_demos/06_calm_soothing.wav${NC}\n"

# Demo 7: Narrator
echo -e "${PURPLE}[7/8] Audiobook Narrator${NC}"
DESC="Professional narrator voice, clear articulation, neutral accent, moderate pacing, consistent delivery."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/07_narrator.wav
echo -e "${GREEN}✓ Saved to voice_demos/07_narrator.wav${NC}\n"

# Demo 8: Elderly Voice
echo -e "${PURPLE}[8/8] Elderly Distinguished${NC}"
DESC="Elderly male voice in his sixties, british accent, distinguished tone, lower pitch, slow and deliberate pacing."
python say.py "$TEST_TEXT" -d "$DESC" -o voice_demos/08_elderly.wav
echo -e "${GREEN}✓ Saved to voice_demos/08_elderly.wav${NC}\n"

echo -e "${CYAN}=========================================${NC}"
echo -e "${GREEN}Demo Complete!${NC}"
echo -e "${CYAN}=========================================${NC}\n"

echo "Generated 8 voice demos in voice_demos/"
echo ""
echo "Listen to the files to hear the differences:"
ls -1 voice_demos/*.wav | nl -w2 -s'. '
echo ""
echo "For more examples, see VOICE_EXAMPLES.md"
echo ""

# Play first example
if command -v play &> /dev/null || command -v aplay &> /dev/null; then
    echo -e "${YELLOW}Playing first example (Professional Male)...${NC}"
    if command -v play &> /dev/null; then
        play voice_demos/01_professional_male.wav 2>/dev/null
    else
        aplay voice_demos/01_professional_male.wav 2>/dev/null
    fi
fi
