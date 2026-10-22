"""Single-rank temporary checkpoint directory without downloading global test datasets."""
import os

import pytest


@pytest.fixture
def tmp_path_dist_ckpt(tmp_path):
    assert int(os.environ.get('WORLD_SIZE', 1)) == 1
    return tmp_path
