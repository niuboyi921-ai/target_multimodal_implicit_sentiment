SENTIMENT_TO_ID = {"positive": 0, "neutral": 1, "negative": 2}
ID_TO_SENTIMENT = {v: k for k, v in SENTIMENT_TO_ID.items()}

BRIDGE_KEYS = (
    "grounded_synthesis",
    "reasoning_transition",
    "evaluative_implication",
)

BRIDGE_SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[GROUND]",
        "[TRANSITION]",
        "[IMPLICATION]",
        "<BRIDGE_BOS>",
    ]
}

GROUND_TOKEN = "[GROUND]"
TRANSITION_TOKEN = "[TRANSITION]"
IMPLICATION_TOKEN = "[IMPLICATION]"
BRIDGE_BOS_TOKEN = "<BRIDGE_BOS>"
# T5's pretrained decoder already knows its native end-of-sequence token.
# Reusing it is both more parameter-efficient and prevents a second EOS concept
# from competing with </s> during bridge generation.
BRIDGE_EOS_TOKEN = "</s>"
