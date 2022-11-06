"""
mtga_ml - a library of machine learning models and tools for analyzing 17lands MTGA data.
"""

import pkgutil

__version__ = (pkgutil.get_data(__package__, "VERSION") or b"").decode("ascii").strip()