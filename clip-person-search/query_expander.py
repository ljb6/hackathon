import anthropic

_client = anthropic.Anthropic()

_SYSTEM = (
    "You convert short person descriptions into a single precise English phrase "
    "for CLIP visual search. Output only the phrase, no explanation.\n"
    "Examples:\n"
    "  bike → a person riding a bicycle\n"
    "  em uma bicicleta → a person riding a bicycle\n"
    "  old man → an elderly man\n"
    "  carrying bags → a person carrying shopping bags\n"
    "  running → a person running\n"
    "  de chapéu → a person wearing a hat\n"
    "If the input is already a good English phrase, return it unchanged."
)


def expand_query(raw: str) -> str:
    if not raw.strip():
        return raw
    try:
        msg = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=60,
            system=_SYSTEM,
            messages=[{"role": "user", "content": raw.strip()}],
        )
        return msg.content[0].text.strip()
    except Exception:
        return raw
