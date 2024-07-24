import pytest

from generate_linkml_from_pydantic import main


@pytest.mark.xfail(
    reason="Awaiting the translation of Pydantic filed types to complete"
)
def test_smoke():
    """Check that I can run this at all"""
    main(["dandischema.models"])
