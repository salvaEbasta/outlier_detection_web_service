import pytest
from ml_microservice import build_app

@pytest.fixture
def app():
    app = build_app()
    yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()