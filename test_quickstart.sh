#!/bin/bash

# Colors for output
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}Maya1 Quickstart Test${NC}"
echo -e "${YELLOW}This tests the standalone quickstart.py example${NC}"
echo -e ""

echo -e "${CYAN}Running quickstart.py...${NC}"
echo -e "${YELLOW}Note: This will download models on first run (~6GB)${NC}"
echo -e "${YELLOW}This may take several minutes...${NC}"
echo -e ""

python quickstart.py

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Quickstart completed successfully!${NC}"

    if [ -f "output.wav" ]; then
        echo -e "${GREEN}Output file generated: output.wav${NC}"

        echo -e "${CYAN}Playing the generated audio...${NC}"
        python3 -c "
import soundfile as sf
import sounddevice as sd
data, samplerate = sf.read('output.wav')
sd.play(data, samplerate)
sd.wait()
"

        echo -e "\n${YELLOW}Press Enter to delete output.wav or Ctrl+C to keep it${NC}"
        read
        rm -f output.wav
        echo -e "${GREEN}Cleaned up output.wav${NC}"
    else
        echo -e "${RED}Warning: output.wav was not generated${NC}"
    fi

    exit 0
else
    echo -e "\n${RED}Quickstart failed!${NC}"
    exit 1
fi
