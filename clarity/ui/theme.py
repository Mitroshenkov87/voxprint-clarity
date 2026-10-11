"""Suite theme from ``voxprint_common.ui.theme``, plus Clarity's splash identity.

The window uses the shared yellow primary actions and soft blue accents.
The splash progress bar is not one of those tokens. ``IDENTITY`` is Clarity's
own colour, passed to ``voxprint_common.ui.splash.Splash``.
"""
from __future__ import annotations

from pathlib import Path

from voxprint_common.ui.theme import (
    ACCENT,
    ACCENT_HOVER,
    ACCENT_SOFT,
    ACCENT_STRONG,
    AMBER,
    BADGE_AMBER_TEXT,
    BADGE_GREEN_TEXT,
    CARD_GLASS,
    CARD_GLASS_MORE,
    CARD_PLAIN,
    CONTRAST_PAIRS,
    CONTROL_BG,
    CONTROL_BORDER,
    CONTROL_HOVER,
    GREEN,
    MENU_BG,
    MENU_SELECTION,
    NOTE_BG,
    NOTE_BORDER,
    PRIMARY,
    PRIMARY_BORDER,
    PRIMARY_DISABLED_BG,
    PRIMARY_DISABLED_TEXT,
    PRIMARY_HOVER,
    RED,
    ROOT_GLASS,
    ROOT_GLASS_MORE,
    ROOT_PLAIN,
    TEXT,
    TEXT_DISABLED,
    TEXT_FAINT,
    TEXT_MUTED,
    TEXT_ON_ACCENT,
    TEXT_ON_PRIMARY,
    build_style as suite_style,
    contrast_ratio,
)

from clarity.appinfo import repo_root

# Splash progress fill. PALETTE.md has no cyan or teal. GREEN is the clear
# accent no other app uses as its splash identity.
IDENTITY = GREEN


def splash_art() -> Path:
    """Return the approved splash image, ``assets/splash/clarity.jpg``.

    Returns:
        The path under the checkout or install root. A missing file is still
        returned; the shared splash paints ``#0b1020`` in that case.
    """
    return repo_root() / "assets" / "splash" / "clarity.jpg"


__all__ = [
    "ACCENT",
    "ACCENT_HOVER",
    "ACCENT_SOFT",
    "ACCENT_STRONG",
    "AMBER",
    "BADGE_AMBER_TEXT",
    "BADGE_GREEN_TEXT",
    "CARD_GLASS",
    "CARD_GLASS_MORE",
    "CARD_PLAIN",
    "CONTRAST_PAIRS",
    "CONTROL_BG",
    "CONTROL_BORDER",
    "CONTROL_HOVER",
    "GREEN",
    "IDENTITY",
    "MENU_BG",
    "MENU_SELECTION",
    "NOTE_BG",
    "NOTE_BORDER",
    "PRIMARY",
    "PRIMARY_BORDER",
    "PRIMARY_DISABLED_BG",
    "PRIMARY_DISABLED_TEXT",
    "PRIMARY_HOVER",
    "RED",
    "ROOT_GLASS",
    "ROOT_GLASS_MORE",
    "ROOT_PLAIN",
    "TEXT",
    "TEXT_DISABLED",
    "TEXT_FAINT",
    "TEXT_MUTED",
    "TEXT_ON_ACCENT",
    "TEXT_ON_PRIMARY",
    "build_style",
    "contrast_ratio",
    "splash_art",
]


def build_style(glass: bool = False, *, more: bool = False) -> str:
    """Return the suite stylesheet plus Clarity's group boxes and Start button.

    Args:
        glass: Use the translucent Acrylic backgrounds from the shared theme.
            The default stays solid. Windows still asks DWM for Acrylic.
        more: Use the more-transparent Acrylic backgrounds. Ignored when
            ``glass`` is false.

    Returns:
        A Qt stylesheet. Qt is not imported.
    """
    rules = f"""
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
    QLabel#faint {{
        color: {TEXT_FAINT};
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
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
    }}
    """
    if not glass:
        rules = f"""
        QWidget {{
            background: {ROOT_PLAIN};
            color: {TEXT};
        }}
        {rules}
        """
    return suite_style(glass, more=more) + rules
