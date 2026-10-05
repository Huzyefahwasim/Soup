"""A cached file's length is not checkpoint provenance."""
from pathlib import Path
import tempfile

import pytest

from scripts.train_dpo import materialize_checkpoint, file_sha256


def test_same_size_wrong_cache_is_replaced():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source, target = root/'source', root/'target'
        source.mkdir()
        target.mkdir()
        (source/'weights').write_bytes(b'correct')
        (target/'weights').write_bytes(b'corrupt')
        hashes = materialize_checkpoint(source, target)
        assert (target/'weights').read_bytes() == b'correct'
        assert hashes['weights'] == file_sha256(source/'weights')


def test_unexpected_cached_weights_are_rejected():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source, target = root/'source', root/'target'
        source.mkdir()
        target.mkdir()
        (source/'config.json').write_text('{}')
        (target/'extra.safetensors').write_bytes(b'wrong')
        with pytest.raises(ValueError, match='Unexpected files'):
            materialize_checkpoint(source, target)
