from pydantic2linkml.cli import main


def test_smoke():
    """Check that I can run this at all"""
    main(["dandischema.models"])
