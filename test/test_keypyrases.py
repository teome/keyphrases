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
    keyphrases.keyphrase_matrix.sort()
    assert np.array_equal(keyphrases.keyphrase_matrix, np.array([28, 28, 35]))


def test_filter_by_frequency_shapes():
    """Ensure we maintain correct dimensions"""
    keyphrases = Keyphrases(FILEPATTERN)
    keyphrases.filter_by_frequency(6, 2)
    assert keyphrases._per_doc_keyphrase_matrix.shape == (2, 756)
    assert keyphrases.keyphrase_matrix.shape == (64,)
    assert keyphrases.feature_names.shape == (64,)


def test_match_sentences(caplog):
    caplog.set_level(logging.DEBUG)
    keyphrases = Keyphrases(FILEPATTERN, 28, 2)
    keyphrases.filter_by_frequency()
    matches = keyphrases.match_sentences()
    assert len(matches) == len(keyphrases.feature_names)
    assert sorted(list(matches.keys())) == sorted(list(keyphrases.feature_names))


def test_render(caplog):
    caplog.set_level(logging.INFO)
    keyphrases = Keyphrases(FILEPATTERN, 2, 2, "<J.*>*<N.*>+")
    keyphrases.filter_by_frequency()
    matches = keyphrases.match_sentences()
    keyphrases.render()


# def test_nomatch():
#     """Ensure we return with nothing but raise no errors"""
#     raise NotImplementedError
