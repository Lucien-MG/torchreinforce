import torchreinforce
from torchreinforce import __version__

def test_version():
    assert __version__ == '0.1.0'

def test_import_agents():
    assert 'agents' in dir(torchreinforce)

def test_import_io():
    assert 'agents' in dir(torchreinforce)
