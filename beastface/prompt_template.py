"""Prompt template parser/renderer.

The user-facing JSON file contains a `prompt`, a `negative_prompt`, and
extra keys whose values are lists of strings. The prompt may reference
those keys with `$name`. Rendering picks one value per variable at random
and substitutes.

The example file uses `# ...` line comments and occasionally relies on a
trailing `,` that ends up inside such a comment. We tolerate both by
stripping comments outside strings and reinserting any missing commas
between adjacent quoted lines before handing off to `json.loads`.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

VAR_RE = re.compile(r'\$([A-Za-z_]\w*)')


def _strip_comments(text: str) -> str:
    out = []
    i = 0
    n = len(text)
    string_char = None
    while i < n:
        c = text[i]
        if string_char is not None:
            out.append(c)
            if c == '\\' and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == string_char:
                string_char = None
            i += 1
            continue
        if c == '"' or c == "'":
            string_char = c
            out.append(c)
            i += 1
            continue
        if c == '#' or (c == '/' and i + 1 < n and text[i + 1] == '/'):
            while i < n and text[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and text[i + 1] == '*':
            i += 2
            while i + 1 < n and not (text[i] == '*' and text[i + 1] == '/'):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def load_template(path: str | Path) -> dict:
    text = Path(path).read_text(encoding='utf-8')
    cleaned = _strip_comments(text)
    # Re-insert any comma that fell off the end of a value because the user
    # tucked it inside the trailing comment.
    cleaned = re.sub(r'("\s*)\n(\s*")', r'\1,\n\2', cleaned)
    cleaned = re.sub(r'(\]\s*)\n(\s*")', r'\1,\n\2', cleaned)
    # And remove any trailing comma before a closer (JSON5-style tolerance).
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    return json.loads(cleaned)


def render(template: dict, rng: random.Random | None = None) -> tuple[str, str]:
    """Return (positive_prompt, negative_prompt) with $vars substituted."""
    rng = rng or random.Random()
    prompt = template.get('prompt', '')
    negative = template.get('negative_prompt', '')

    def repl(match):
        name = match.group(1)
        choices = template.get(name)
        if not isinstance(choices, list) or not choices:
            return match.group(0)
        return str(rng.choice(choices))

    return VAR_RE.sub(repl, prompt), VAR_RE.sub(repl, negative)
