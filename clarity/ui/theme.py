"""Suite palette: yellow primary actions, soft blue accents, dark surfaces.

Tokens match the Voxprint palette (Audiobook Builder is the source of truth).
Text colours are opaque. Contrast pairs are checked in tests.
"""
from __future__ import annotations

PRIMARY = "#f5d76e"
PRIMARY_HOVER = "#f9e291"
PRIMARY_BORDER = "#fbe8a6"
TEXT_ON_PRIMARY = "#1c1604"
PRIMARY_DISABLED_BG = "#36321f"
PRIMARY_DISABLED_TEXT = "#b9b08a"

MENU_BG = "#1c2535"
MENU_SELECTION = "#2f4c78"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"
TEXT_ON_ACCENT = "#0b1220"

TEXT = "#f2f2f5"
TEXT_MUTED = "#c4c4d0"
TEXT_FAINT = "#a8a8b8"
TEXT_DISABLED = "#8e8e9e"

ROOT_PLAIN = "#17171c"
CARD_PLAIN = "#202029"
CONTROL_BG = "#2c2c38"
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"

AMBER = "#fcd34d"
NOTE_BG = "#2a2418"
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
BADGE_GREEN_TEXT = "#0b1f12"
BADGE_AMBER_TEXT = "#251a02"
RED = "#fca5a5"

# Clarity's splash identity. Every other control uses the shared tokens.
SPLASH = "#f5d76e"
WHITE = "#ffffff"


def _channel(hex_color: str) -> tuple[int, int, int]:
    text = hex_color.removeprefix("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_color: str) -> float:
    red, green, blue = _channel(hex_color)
    return 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)


def contrast_ratio(foreground: str, background: str) -> float:
    lighter = max(luminance(foreground), luminance(background))
    darker = min(luminance(foreground), luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


# Opaque pairs that must clear WCAG AA for normal text.
CONTRAST_PAIRS: tuple[tuple[str, str], ...] = (
    (TEXT, ROOT_PLAIN),
    (TEXT, CARD_PLAIN),
    (TEXT, CONTROL_BG),
    (TEXT, MENU_BG),
    (TEXT_MUTED, ROOT_PLAIN),
    (TEXT_FAINT, ROOT_PLAIN),
    (TEXT_ON_PRIMARY, PRIMARY),
    (WHITE, MENU_SELECTION),
    (WHITE, ACCENT_STRONG),
    (BADGE_GREEN_TEXT, GREEN),
    (BADGE_AMBER_TEXT, AMBER),
    (TEXT_ON_ACCENT, ACCENT_SOFT),
)


def build_style() -> str:
    """Qt stylesheet. The window fill is the plain dark token (Acrylic tints the frame on Windows)."""
    return f"""
    QWidget {{
        background: {ROOT_PLAIN};
        color: {TEXT};
        font-size: 14px;
    }}
    QLabel#subtitle, QLabel#hint {{
        color: {TEXT_MUTED};
    }}
    QLabel#faint {{
        color: {TEXT_FAINT};
    }}
    QGroupBox {{
        background: {CARD_PLAIN};
        border: 1px solid {CONTROL_BORDER};
        border-radius: 10px;
        margin-top: 12px;
        padding: 12px;
    }}
    QGroupBox::title {{
        color: {TEXT};
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {CONTROL_BG};
        color: {TEXT};
        border: 1px solid {CONTROL_BORDER};
        border-radius: 6px;
        padding: 6px;
        min-height: 22px;
    }}
    QComboBox QAbstractItemView {{
        background: {MENU_BG};
        color: {TEXT};
        selection-background-color: {MENU_SELECTION};
        selection-color: {WHITE};
    }}
    QPushButton {{
        background: {CONTROL_BG};
        color: {TEXT};
        border: 1px solid {CONTROL_BORDER};
        border-radius: 8px;
        padding: 8px 14px;
    }}
    QPushButton:hover {{
        background: {CONTROL_HOVER};
    }}
    QPushButton#start {{
        background: {PRIMARY};
        color: {TEXT_ON_PRIMARY};
        border: 1px solid {PRIMARY_BORDER};
        font-weight: 600;
    }}
    QPushButton#start:hover {{
        background: {PRIMARY_HOVER};
    }}
    QPushButton#start:disabled {{
        background: {PRIMARY_DISABLED_BG};
        color: {PRIMARY_DISABLED_TEXT};
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
    }}
    QSlider::groove:horizontal {{
        background: {CONTROL_BG};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QProgressBar {{
        background: {CONTROL_BG};
        border: 1px solid {CONTROL_BORDER};
        border-radius: 6px;
        text-align: center;
        color: {TEXT};
    }}
    QProgressBar::chunk {{
        background: {ACCENT_STRONG};
        border-radius: 5px;
    }}
    QPlainTextEdit {{
        background: {CARD_PLAIN};
        color: {TEXT};
        border: 1px solid {CONTROL_BORDER};
        border-radius: 8px;
    }}
    """
