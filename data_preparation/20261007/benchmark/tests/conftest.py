"""Default to lightweight unit tests; opt into scientific integration tests remotely."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--run-integration', action='store_true', default=False)


def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: real scVelo geometry/postprocess (remote opt-in)')


def pytest_collection_modifyitems(config, items):
    if not config.getoption('--run-integration'):
        for item in items:
            if 'integration' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='remote opt-in: --run-integration'))
