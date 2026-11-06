import hashlib

import pytest

from fraud_api.data import FILE_NAME, ChecksumMismatchError, download

CONTENT = b"fraud_bool,month\n0,0\n"
CONTENT_SHA256 = hashlib.sha256(CONTENT).hexdigest()


def _fetch_writing(content):
    calls = []

    def fetch(directory):
        calls.append(directory)
        (directory / FILE_NAME).write_bytes(content)

    return fetch, calls


def test_downloads_and_verifies_the_file(tmp_path):
    fetch, calls = _fetch_writing(CONTENT)

    path = download(tmp_path, fetch=fetch, expected_sha256=CONTENT_SHA256)

    assert path.read_bytes() == CONTENT
    assert len(calls) == 1


def test_rejects_a_corrupted_download_and_deletes_it(tmp_path):
    fetch, _ = _fetch_writing(CONTENT + b"corrupted")

    with pytest.raises(ChecksumMismatchError):
        download(tmp_path, fetch=fetch, expected_sha256=CONTENT_SHA256)

    assert not (tmp_path / FILE_NAME).exists()


def test_skips_the_download_when_a_verified_copy_exists(tmp_path):
    (tmp_path / FILE_NAME).write_bytes(CONTENT)
    fetch, calls = _fetch_writing(b"never used")

    download(tmp_path, fetch=fetch, expected_sha256=CONTENT_SHA256)

    assert calls == []


def test_replaces_an_existing_corrupted_copy(tmp_path):
    (tmp_path / FILE_NAME).write_bytes(b"truncated")
    fetch, calls = _fetch_writing(CONTENT)

    path = download(tmp_path, fetch=fetch, expected_sha256=CONTENT_SHA256)

    assert path.read_bytes() == CONTENT
    assert len(calls) == 1


def test_a_download_that_fails_partway_leaves_nothing_behind(tmp_path):
    def fetch(directory):
        (directory / FILE_NAME).write_bytes(CONTENT[:5])
        raise ConnectionError("network dropped")

    with pytest.raises(ConnectionError):
        download(tmp_path, fetch=fetch, expected_sha256=CONTENT_SHA256)

    assert not (tmp_path / FILE_NAME).exists()
