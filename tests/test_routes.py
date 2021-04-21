import pytest

@pytest.mark.filterwarnings("ignore:.*U.*mode is deprecated:DeprecationWarning")
def test_models_get(client):
    r = client.get('/models')
    print(r.get_json())
    assert r.get_json().get('available', None) != None
