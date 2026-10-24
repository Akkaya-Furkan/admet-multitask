import hashlib

import pytest

from scripts import download_data


def test_download_is_verified_and_reread(tmp_path, monkeypatch):
    raw = b"Drug\tY\nCCO\t1\n"
    monkeypatch.setattr(download_data, "SOURCES", {"test": 1})
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", {"test": hashlib.sha256(raw).hexdigest()})
    class Response:
        content = raw
        def raise_for_status(self):
            pass
    monkeypatch.setattr(download_data.requests, "get", lambda *a, **k: Response())
    first = download_data.download(tmp_path)
    assert (tmp_path / "test.tab").read_bytes() == raw
    assert first["test"]["rows"] == 1
    assert first == download_data.download(tmp_path)
    (tmp_path / "test.tab").write_bytes(raw + b"CCC\t0\n")
    with pytest.raises(ValueError, match="checksum"):
        download_data.download(tmp_path)


def test_changed_remote_data_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "SOURCES", {"test": 1})
    monkeypatch.setattr(download_data, "EXPECTED_SHA256", {"test": "wrong"})
    class Response:
        content = b"Drug\tY\nCCO\t1\n"
        def raise_for_status(self):
            pass
    monkeypatch.setattr(download_data.requests, "get", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="checksum"):
        download_data.download(tmp_path)
    assert list(tmp_path.iterdir()) == []
