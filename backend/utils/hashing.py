import hashlib
from pathlib import Path


def generate_hash(path: Path) -> str:
    sha = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(8192):
            sha.update(chunk)

    return sha.hexdigest()