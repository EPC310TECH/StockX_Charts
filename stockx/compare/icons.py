import re
from pathlib import Path

ICONS_DIR = Path(__file__).resolve().parent / "icons"


def load_icon(name: str, size: int = 16) -> str:
    """Raw <svg> markup for a lucide-static icon (fetched from
    https://unpkg.com/lucide-static, MIT licensed), resized from its
    default 24x24. Icons use stroke="currentColor" by convention, so they
    inherit whatever CSS `color` is set on their containing element --
    no per-color variants needed, just wrap in a span/div with the color
    you want."""
    svg = (ICONS_DIR / f"{name}.svg").read_text()
    svg = re.sub(r'width="\d+"', f'width="{size}"', svg, count=1)
    svg = re.sub(r'height="\d+"', f'height="{size}"', svg, count=1)
    return svg
