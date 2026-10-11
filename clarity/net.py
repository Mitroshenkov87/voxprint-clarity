"""HTTPS downloads trust the operating-system certificate store.

Python's default store rejected the archive.org redirect on the RTX 4090 VM
(``CERTIFICATE_VERIFY_FAILED``, a self-signed chain). ``curl.exe`` succeeded
because it uses the Windows store. ``truststore`` makes ``urllib`` do the same.
"""
from __future__ import annotations

import truststore

_INSTALLED = False


def install_system_trust() -> None:
    """Point the default SSL context at the system certificate store.

    Safe to call more than once. Later ``urlopen`` calls, including ones inside
    ``voxprint_engine``, pick up the injected context.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    truststore.inject_into_ssl()
    _INSTALLED = True
