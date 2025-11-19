#!/bin/bash

# Colors for output
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

LONG_TEXT="This is a very long piece of text that will take some time to speak. It contains multiple sentences and should take at least 10 seconds to say. Let me tell you about text to speech systems. They convert written text into spoken words. This technology has many applications in accessibility, education, and entertainment. The quality of synthetic speech has improved dramatically over the years."

wait_key() {
    echo -e "\n${YELLOW}Press Enter to continue...${NC}"
    read
}

play_file() {
    local file=$1
    echo -e "${CYAN}Playing file: $file${NC}"
    python3 -c "
import soundfile as sf
import sounddevice as sd
data, samplerate = sf.read('$file')
sd.play(data, samplerate)
sd.wait()
"
}

run_test() {
    local test_num=$1
    local description=$2
    shift 2
    echo -e "\n${CYAN}Test ${test_num}: ${description}${NC}"
    echo -e "${GREEN}Running: $@${NC}"
    "$@"
}

echo -e "${CYAN}Maya1 TTS Tests${NC}"

echo -e "\n${CYAN}Basic Synthesis Tests${NC}"

run_test "1.1" "Basic synthesis with default voice" \
    python say.py "Test 1.1: This is a basic synthesis test with the default voice"
wait_key

run_test "1.2" "Custom voice description - young female" \
    python say.py -d "Young female voice, excited, high pitch" "Test 1.2: Testing with a young female voice"
wait_key

run_test "1.3" "Custom voice description - elderly male" \
    python say.py -d "Elderly male voice, british accent, slow pacing" "Test 1.3: Testing with an elderly British male voice"
wait_key

echo -e "\n${CYAN}Emotion Tag Tests${NC}"

run_test "2.1" "Laugh emotion" \
    python say.py "Test 2.1: This is funny! <laugh> Really funny!"
wait_key

run_test "2.2" "Multiple emotions" \
    python say.py "Test 2.2: <whisper> This is a secret. <laugh_harder> Just kidding!"
wait_key

run_test "2.3" "Sigh and gasp" \
    python say.py "Test 2.3: I'm so tired <sigh> Oh no! <gasp>"
wait_key

echo -e "\n${CYAN}File Output Tests${NC}"

run_test "3.1" "Basic file output" \
    python say.py --output test3_1.wav "Test 3.1: This is a basic file output test"
wait_key
play_file "test3_1.wav"
wait_key

run_test "3.2" "File output with custom voice" \
    python say.py -d "Deep male voice, warm, slow" --output test3_2.wav "Test 3.2: This is a file output with a deep male voice"
wait_key
play_file "test3_2.wav"
wait_key

run_test "3.3" "File output with emotions" \
    python say.py --output test3_3.wav "Test 3.3: Hello! <laugh> This has emotions! <giggle>"
wait_key
play_file "test3_3.wav"
wait_key

echo -e "\n${CYAN}Parameter Adjustment Tests${NC}"

run_test "4.1" "Higher temperature (more creative)" \
    python say.py --temperature 0.7 "Test 4.1: Testing with higher temperature for more variation"
wait_key

run_test "4.2" "Lower temperature (more consistent)" \
    python say.py --temperature 0.2 "Test 4.2: Testing with lower temperature for consistency"
wait_key

run_test "4.3" "More tokens for longer generation" \
    python say.py --max-tokens 3000 "Test 4.3: ${LONG_TEXT}"
wait_key

echo -e "\n${CYAN}Long Text Tests${NC}"

run_test "5.1" "Long text with default settings" \
    python say.py "${LONG_TEXT}"
wait_key

run_test "5.2" "Long text to file" \
    python say.py --output test5_2.wav "${LONG_TEXT}"
wait_key
play_file "test5_2.wav"
wait_key

echo -e "\n${CYAN}Voice Description Variety Tests${NC}"

run_test "6.1" "American accent, conversational" \
    python say.py -d "Male voice, 40s, american accent, conversational" "Test 6.1: American conversational voice"
wait_key

run_test "6.2" "British accent, formal" \
    python say.py -d "Female voice, 50s, british accent, formal, clear" "Test 6.2: British formal voice"
wait_key

run_test "6.3" "Energetic and upbeat" \
    python say.py -d "Young female, energetic, upbeat, fast pacing" "Test 6.3: This is an energetic and upbeat voice!"
wait_key

echo -e "\n${CYAN}Verbose Mode Test${NC}"

run_test "7.1" "Verbose output" \
    python say.py -v "Test 7.1: Testing verbose mode to see detailed logging"
wait_key

echo -e "\n${CYAN}Server Check Test${NC}"

run_test "8.1" "Server status check" \
    python say.py --server-check
wait_key

# Clean up files
rm -f test*.wav

echo -e "\n${GREEN}Tests complete!${NC}"
echo -e "${YELLOW}Note: Maya1 uses natural language descriptions instead of predefined voices${NC}"
echo -e "${YELLOW}You can create any voice by describing it in natural language!${NC}"
