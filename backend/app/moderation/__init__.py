import re
import unicodedata

THREAT = re.compile(
    r"\b(?:i(?:'m| am) going to|i(?:'ll| will)|i want to|we(?:'ll| will)) "
    r"(?:hurt|kill|shoot|stab|attack)\s+(?:you|him|her|them|(?:the |your )?(?:agent|seller|owner))\b"
)
ABUSE = re.compile(
    r"\b(?:fuck you|"
    r"you (?:are |are an? )?(?:idiot|fool|moron|stupid)|"
    r"you(?:'re| are) (?:an? )?(?:idiot|fool|moron|stupid)|"
    r"are you (?:(?:completely|totally) )?out of your mind|"
    r"are you (?:crazy|mad|stupid)|"
    r"what(?:'s| is) wrong with you|shut up)\b"
)
PROFANITY = re.compile(r"\b(?:damn|shit|fuck(?:ing)?|bullshit)\b")


def moderate(message: str) -> str:
    normalized = unicodedata.normalize("NFKC", message).casefold().replace("’", "'")
    normalized = " ".join(normalized.split())
    normalized = re.sub(r"\bf[\*u]ck\b", "fuck", normalized)
    normalized = re.sub(r"\bsh[1i]t\b", "shit", normalized)
    if THREAT.search(normalized):
        return "threat"
    if ABUSE.search(normalized):
        return "directed_abuse"
    if PROFANITY.search(normalized):
        return "mild_profanity"
    return "clean"
