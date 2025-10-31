import re
from typing import List


_punct_re = re.compile(r"[^\w\s]", flags=re.UNICODE)


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = _punct_re.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def batchify(items: List[str], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

