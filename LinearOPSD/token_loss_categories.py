import re
import string


CATEGORY_ORDER = [
    "reasoning_discourse_words",
    "answer_finalization",
    "math_symbols",
    "numbers",
    "punctuation_formatting",
    "math_action_words",
    "math_object_words",
    "proof_structure_words",
    "special_tokens",
    "ordinary_words",
]


REASONING_DISCOURSE_WORDS = {
    "maybe",
    "perhaps",
    "probably",
    "possibly",
    "let",
    "okay",
    "ok",
    "alright",
    "hmm",
    "wait",
    "because",
    "since",
    "so",
    "thus",
    "hence",
    "therefore",
    "but",
    "however",
    "although",
    "though",
    "yet",
    "or",
    "alternatively",
    "instead",
    "otherwise",
    "actually",
    "really",
    "just",
    "simply",
    "basically",
    "very",
    "quite",
    "pretty",
    "rather",
    "fairly",
    "now",
    "then",
    "next",
    "first",
    "second",
    "finally",
    "try",
    "see",
    "check",
    "note",
    "recall",
    "think",
    "idea",
    "strategy",
    "approach",
    "method",
    "way",
    "would",
    "could",
    "should",
    "might",
    "can",
    "huge",
    "large",
    "big",
    "small",
    "tiny",
    "interesting",
    "tricky",
    "complex",
    "simple",
}


ANSWER_FINALIZATION_WORDS = {
    "answer",
    "final",
    "result",
    "conclusion",
    "conclude",
    "solution",
    "boxed",
}


MATH_ACTION_WORDS = {
    "add",
    "calculate",
    "compare",
    "compute",
    "count",
    "determine",
    "divide",
    "evaluate",
    "expand",
    "factor",
    "find",
    "multiply",
    "rearrange",
    "simplify",
    "solve",
    "square",
    "subtract",
    "substitute",
}


MATH_OBJECT_WORDS = {
    "angle",
    "circle",
    "coefficient",
    "equation",
    "expression",
    "factor",
    "fraction",
    "function",
    "graph",
    "integer",
    "multiple",
    "number",
    "probability",
    "ratio",
    "set",
    "square",
    "term",
    "triangle",
    "variable",
}


PROOF_STRUCTURE_WORDS = {
    "assume",
    "base",
    "case",
    "condition",
    "contradiction",
    "define",
    "equivalent",
    "follows",
    "given",
    "implies",
    "induction",
    "satisfies",
    "suppose",
}


MATH_SYMBOL_PATTERNS = (
    "\\",
    "^",
    "_",
    "{",
    "}",
    "[",
    "]",
    "(",
    ")",
    "=",
    "+",
    "-",
    "*",
    "/",
    "<",
    ">",
    "|",
    "\u00b1",
    "\u2264",
    "\u2265",
    "\u2260",
    "\u00d7",
    "\u00f7",
    "\u221a",
    "\u2211",
    "\u222b",
    "\u03c0",
)


PUNCTUATION_CHARS = set(string.punctuation) - set("\\^_{}[]=+-*/<>|")


class TokenCategoryClassifier:
    """Classify decoded tokenizer tokens into coarse diagnostic categories."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        self._cache = {}

    def category_for_token_id(self, token_id):
        token_id = int(token_id)
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached

        token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
        category = self.category_for_text(token_text, token_id=token_id)
        self._cache[token_id] = category
        return category

    def category_for_text(self, token_text, token_id=None):
        if token_id is not None and int(token_id) in self.special_token_ids:
            return "special_tokens"

        text = token_text or ""
        stripped = text.strip()
        if not stripped:
            return "punctuation_formatting"
        if stripped.startswith("<|") and stripped.endswith("|>"):
            return "special_tokens"

        word = _single_word(stripped)
        if word in REASONING_DISCOURSE_WORDS:
            return "reasoning_discourse_words"
        if word in ANSWER_FINALIZATION_WORDS:
            return "answer_finalization"
        if _is_number_token(stripped):
            return "numbers"
        if _is_math_symbol_token(stripped):
            return "math_symbols"
        if _is_punctuation_formatting_token(stripped):
            return "punctuation_formatting"
        if word in MATH_ACTION_WORDS:
            return "math_action_words"
        if word in MATH_OBJECT_WORDS:
            return "math_object_words"
        if word in PROOF_STRUCTURE_WORDS:
            return "proof_structure_words"
        return "ordinary_words"


def _single_word(text):
    words = re.findall(r"[A-Za-z]+", text.lower())
    if len(words) != 1:
        return None
    return words[0]


def _is_number_token(text):
    return any(ch.isdigit() for ch in text) and all(
        ch.isdigit() or ch.isspace() or ch in ".,:%+-/^_{}()[]$" for ch in text
    )


def _is_math_symbol_token(text):
    if any(pattern in text for pattern in MATH_SYMBOL_PATTERNS):
        return True
    if text.startswith("$") or text.endswith("$"):
        return True
    return False


def _is_punctuation_formatting_token(text):
    return all(ch.isspace() or ch in PUNCTUATION_CHARS or ch == "#" for ch in text)
