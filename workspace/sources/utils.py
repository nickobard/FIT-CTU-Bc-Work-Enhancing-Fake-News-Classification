import os

def generate_random_state():
    return int.from_bytes(os.urandom(4), byteorder="big")