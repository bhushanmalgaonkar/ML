from functools import partial
from os import path


def number_of_lines(filepath):
    if not path.exists(filepath):
        return 0

    buffer = 2**16
    with open(filepath) as f:
        return sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))


def get_last_line(filepath):
    if not path.exists(filepath):
        return None

    last_line = None
    with open(filepath) as f:
        for line in f:
            last_line = line
    return last_line
