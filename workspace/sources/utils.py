import os
from urllib.parse import urlparse
from urllib.request import url2pathname


def generate_random_state():
    return int.from_bytes(os.urandom(4), byteorder="big")


def get_normalized_path_from_artifact_uri(artifact_uri: str) -> str:
    parsed_path = urlparse(artifact_uri).path
    local_path = url2pathname(parsed_path)
    normalized_path = os.path.normpath(local_path)
    return normalized_path

