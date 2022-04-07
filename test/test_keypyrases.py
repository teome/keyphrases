from glob import glob
import os
import pytest

from keyphrases.keyphrases import Keyphrases

TEXTFILE_ROOT = "./data"
FILEPATTERN = os.path.join(TEXTFILE_ROOT, "*[12].txt")


def test_todo():
    keyphrases = Keyphrases(FILEPATTERN)
    keyphrases._filter_by_frequency(5, 2)
