from enterprise_ai import __version__


def test_version():
    assert __version__ == "0.1.0", f"Expected version 0.1.0 but got {__version__}"
