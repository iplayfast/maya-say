#!/usr/bin/env python3
"""
Utility functions for Maya1 TTS processing.
"""

from typing import List
from src.constants import (
    CODE_START_TOKEN_ID,
    CODE_END_TOKEN_ID,
    CODE_TOKEN_OFFSET,
    SNAC_MIN_ID,
    SNAC_MAX_ID,
    SNAC_TOKENS_PER_FRAME,
    SOH_ID,
    EOH_ID,
    SOA_ID,
    BOS_ID,
    TEXT_EOT_ID,
)


def build_prompt(tokenizer, description: str, text: str) -> str:
    """
    Build formatted prompt for Maya1.
    Uses manual format to avoid system messages from chat template.

    Args:
        tokenizer: The Maya1 tokenizer
        description: Natural language voice description
        text: Text to synthesize (may include emotion tags)

    Returns:
        Formatted prompt string
    """
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    # Ensure description ends with period (significantly reduces vocalization)
    # Community-discovered fix: https://github.com/ysharma3501/FastMaya
    if description and not description.rstrip().endswith('.'):
        description = description.rstrip() + '.'

    # Format: <description="..."> text
    formatted_text = f'<description="{description}"> {text}'

    # Manual prompt structure (avoids chat template's system message)
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )

    return prompt


def extract_snac_codes(token_ids: List[int]) -> List[int]:
    """
    Extract SNAC codes from generated tokens.

    Args:
        token_ids: List of generated token IDs

    Returns:
        List of SNAC token IDs
    """
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)

    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]

    return snac_codes


def unpack_snac_from_7(snac_tokens: List[int]) -> List[List[int]]:
    """
    Unpack 7-token SNAC frames to 3 hierarchical levels.

    Args:
        snac_tokens: List of SNAC token IDs (should be multiple of 7)

    Returns:
        List of 3 lists [L1, L2, L3] containing unpacked codes
    """
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i*7:(i+1)*7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])

    return [l1, l2, l3]


def validate_description(description: str) -> bool:
    """
    Validate that a voice description is reasonable.

    Args:
        description: Voice description string

    Returns:
        True if valid, False otherwise
    """
    if not description or len(description.strip()) == 0:
        return False

    # Basic length check
    if len(description) > 500:  # Reasonable max length
        return False

    return True
