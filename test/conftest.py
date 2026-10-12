import pytest


@pytest.fixture(autouse=True)
def _counters_off_after_each_test():
    """A test that counted blocks (_testing.reset_counters) must not leave counting on for the
    tests after it: decoding threads would contend for the counters' cache lines."""
    yield
    try:
        from bslz4_to_sparse import _testing
    except Exception:  # the package is not importable in this environment: nothing to switch off
        return
    _testing.stop_counters()
