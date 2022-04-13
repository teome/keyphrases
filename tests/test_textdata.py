from glob import glob
import os
import pytest

from keyphrases.textdata import TextData

TEXTFILE_ROOT = "./data"


@pytest.fixture
def textdata():
    return TextData(os.path.join(TEXTFILE_ROOT, "*.txt"))


def test_file_load(textdata):
    # textdata = TextData(os.path.join(TEXTFILE_ROOT, "*.txt"))

    assert (
        os.path.join(TEXTFILE_ROOT, "doc1.txt") in textdata.filenames
    ), "failed to load files and access filenames"


def test_iterator(textdata):
    assert len([text for text in textdata]) == len(
        list(glob(os.path.join(TEXTFILE_ROOT, "*.txt")))
    )
