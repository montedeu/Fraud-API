"""Download the BAF ``Base.csv`` dataset and verify it."""

import hashlib
from collections.abc import Callable
from pathlib import Path

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
FILE_NAME = "Base.csv"
SHA256 = "7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809"
DATA_DIR = Path("data")


class ChecksumMismatchError(Exception):
    """The dataset file is not the one the project was built on."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str = SHA256) -> None:
    """Raise ``ChecksumMismatchError`` unless ``path`` has the expected SHA-256."""
    actual = sha256(path)
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"{path} has SHA-256 {actual}, expected {expected_sha256}. "
            "Delete it and download it again."
        )


def download(
    directory: Path = DATA_DIR,
    *,
    fetch: Callable[[Path], None] | None = None,
    expected_sha256: str = SHA256,
) -> Path:
    """Download ``Base.csv`` into ``directory`` unless a verified copy is already there.

    A downloaded file that fails verification is deleted before the error is raised,
    so a corrupted file never stays behind looking usable.
    """
    path = directory / FILE_NAME
    if path.exists():
        verify(path, expected_sha256)
        return path
    directory.mkdir(parents=True, exist_ok=True)
    (fetch or _fetch_from_kaggle)(directory)
    try:
        verify(path, expected_sha256)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _fetch_from_kaggle(directory: Path) -> None:
    # Imported here: the Kaggle client authenticates on import and is only needed
    # for this one-off download, never by tests or the service.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(
        KAGGLE_DATASET, FILE_NAME, path=str(directory), quiet=False, unzip=True
    )
