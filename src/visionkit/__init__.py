"""VisionKit: Compatibility alias for AniVis.

Basic usage:
------------
    >>> import visionkit as vk
    >>> image = vk.read("input.jpg")
"""

from anivis import *  # noqa: F401, F403
from anivis import __all__, __version__  # noqa: F401
