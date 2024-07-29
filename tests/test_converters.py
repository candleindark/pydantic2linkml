import pytest

from generate_linkml_from_pydantic import main


def test_smoke():
    """Check that I can run this at all"""
    main(["dandischema.models"])
