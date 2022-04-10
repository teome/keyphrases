from glob import glob
import logging
import os

import numpy as np
import pytest

from keyphrases.keyphrases import Keyphrases

TEXTFILE_ROOT = "./data"
FILEPATTERN = os.path.join(TEXTFILE_ROOT, "*[12].txt")


def test_filter_by_frequency():
    keyphrases = Keyphrases(FILEPATTERN)
    keyphrases.filter_by_frequency(28, 2)
    keyphrases._keyphrase_matrix.sort()
    assert np.array_equal(keyphrases._keyphrase_matrix, np.array([28, 28, 35]))


def test_filter_by_frequency_shapes():
    keyphrases = Keyphrases(FILEPATTERN)
    keyphrases.filter_by_frequency(6, 2)
    assert keyphrases._keyphrase_matrix.min() >= 2


def test_match_sentences(caplog):
    caplog.set_level(logging.DEBUG)
    keyphrases = Keyphrases(FILEPATTERN)


def test_nomatch():
    """Ensure we return with nothing but raise no errors"""
    raise NotImplementedError
