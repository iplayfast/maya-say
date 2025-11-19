#!/bin/bash

# Colors for output
CYAN='\033[1;36m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
PURPLE='\033[1;35m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Maya1 TTS Performance Timing Tests       ${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}\n"

# Test text
SHORT_TEXT="Hello, how are you today?"
MEDIUM_TEXT="This is a medium length sentence with several words to test the performance of the text to speech system."
LONG_TEXT="This is a longer piece of text that contains multiple sentences. The purpose is to measure how the system performs with extended speech synthesis. Text to speech technology has come a long way in recent years, with neural networks providing increasingly natural sounding voices."

# Voice descriptions
FEMALE_ENGLISH="Young female voice, clear, american accent"
MALE_CANADIAN="Middle-aged male voice, canadian accent, warm"
FEMALE_ENGLISH_2="Young female voice, clear, american accent"  # Same as first
MALE_BRITISH="Elderly male voice, british accent, formal"
FEMALE_EXCITED="Young female voice, excited, energetic, fast pacing"

# Kill any existing server
echo -e "${YELLOW}Step 1: Cleaning up existing server...${NC}"
python say.py --kill 2>/dev/null
echo -e "${YELLOW}Waiting for server to fully terminate...${NC}"
sleep 5
# Verify server is actually stopped
if python say.py --server-check 2>/dev/null | grep -q "running"; then
    echo -e "${RED}⚠ Server still running, waiting longer...${NC}"
    sleep 5
fi
echo -e "${GREEN}✓ Server stopped${NC}\n"

# Array to store timing results
declare -a RESULTS

run_timed_test() {
    local test_num=$1
    local description=$2
    local voice_desc=$3
    local text=$4

    echo -e "${CYAN}Test ${test_num}: ${description}${NC}"
    echo -e "${PURPLE}Voice: ${voice_desc}${NC}"
    echo -e "${PURPLE}Text length: ${#text} chars${NC}"

    # Run and time the command
    local start=$(date +%s.%N)
    python say.py -d "$voice_desc" "$text" 2>&1 | grep -E "INFO:|Model server"
    local end=$(date +%s.%N)

    # Calculate duration
    local duration=$(echo "$end - $start" | bc)

    echo -e "${GREEN}⏱  Duration: ${duration}s${NC}\n"

    # Store result
    RESULTS+=("Test $test_num ($description): ${duration}s")

    # Small delay between tests
    sleep 1
}

echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Starting Timing Tests                    ${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}\n"

# Test 1: First request - cold start (server initialization + first synthesis)
run_timed_test "1" "Cold start - Female English (short)" \
    "$FEMALE_ENGLISH" \
    "$SHORT_TEXT"

# Test 2: Second request - warm server, different voice description
run_timed_test "2" "Warm server - Male Canadian (short)" \
    "$MALE_CANADIAN" \
    "$SHORT_TEXT"

# Test 3: Same voice as Test 1 - check if caching helps
run_timed_test "3" "Repeated voice - Female English (short)" \
    "$FEMALE_ENGLISH_2" \
    "$SHORT_TEXT"

# Test 4: New voice with medium text
run_timed_test "4" "New voice - Male British (medium)" \
    "$MALE_BRITISH" \
    "$MEDIUM_TEXT"

# Test 5: Original voice with longer text
run_timed_test "5" "Original voice - Female English (long)" \
    "$FEMALE_ENGLISH" \
    "$LONG_TEXT"

# Test 6: Different text length, same voice
run_timed_test "6" "Same voice - Male Canadian (medium)" \
    "$MALE_CANADIAN" \
    "$MEDIUM_TEXT"

# Test 7: Energetic voice variation
run_timed_test "7" "Variation - Female Excited (short)" \
    "$FEMALE_EXCITED" \
    "$SHORT_TEXT"

# Test 8: Repeat first test to see consistency
run_timed_test "8" "Consistency check - Female English (short)" \
    "$FEMALE_ENGLISH" \
    "$SHORT_TEXT"

# Print summary
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Timing Results Summary                   ${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}\n"

for result in "${RESULTS[@]}"; do
    echo -e "${GREEN}$result${NC}"
done

echo -e "\n${CYAN}════════════════════════════════════════════${NC}"
echo -e "${CYAN}   Analysis Notes                            ${NC}"
echo -e "${CYAN}════════════════════════════════════════════${NC}"
echo -e "${YELLOW}Test 1:${NC} Cold start - includes server startup + model loading"
echo -e "${YELLOW}Test 2:${NC} Warm server, new voice - tests voice switching overhead"
echo -e "${YELLOW}Test 3:${NC} Repeated voice - checks for voice description caching"
echo -e "${YELLOW}Test 4:${NC} Medium text - tests text length impact"
echo -e "${YELLOW}Test 5:${NC} Long text, original voice - tests text scaling"
echo -e "${YELLOW}Test 6:${NC} Same voice, medium text - tests consistency"
echo -e "${YELLOW}Test 7:${NC} Voice variation - tests description complexity"
echo -e "${YELLOW}Test 8:${NC} Consistency check - validates performance stability"

echo -e "\n${CYAN}Key Insights:${NC}"
echo -e "• Compare Test 1 vs Test 2: Server startup overhead"
echo -e "• Compare Test 2 vs Test 3: Voice description caching (if any)"
echo -e "• Compare Test 3 vs Test 8: Performance consistency"
echo -e "• Compare Test 4 vs Test 5: Text length scaling"
echo -e "\n"
