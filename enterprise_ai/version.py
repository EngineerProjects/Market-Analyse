"""
Version information for the Enterprise AI package.

This module follows semantic versioning (SemVer) and PEP 440 guidelines.
"""

from collections import namedtuple  # <-- Move the import to the top

__all__ = ["__version__", "version_info", "VERSION"]

# Version as tuple (major, minor, micro)
VERSION = (0, 1, 0)

# String version representation
__version__ = ".".join(map(str, VERSION))

# Version information as named tuple
VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro"])

version_info = VersionInfo(major=VERSION[0], minor=VERSION[1], micro=VERSION[2])

# Compatibility with standard __version_info__ pattern
__version_info__ = VERSION


def get_version(pretty: bool = False) -> str:
    """Return the package version.

    Args:
        pretty (bool): If True, returns a formatted version string
    Returns:
        str: Version string
    """
    if pretty:
        return f"Enterprise AI v{__version__}"
    return __version__


if __name__ == "__main__":
    # Show version when executed directly
    print(f"Enterprise AI version: {__version__}")
    print(f"Version tuple: {VERSION}")
    print(f"Version info: {version_info}")
