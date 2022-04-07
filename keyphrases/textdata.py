from glob import glob
from typing import Dict, List
import spacy


class TextData:
    def __init__(self, fname_glob_pattern: str):
        text_files = glob(fname_glob_pattern, recursive=True)
        if not text_files:
            raise RuntimeError(
                f"Failed to find text files to load using '{fname_glob_pattern}'"
            )
        self._text_files = text_files
        self._sents = None

    @property
    def filenames(self) -> List[str]:
        return self._text_files

    def __len__(self) -> int:
        return len(self._text_files)

    def __iter__(self) -> str:
        for fname in self._text_files:
            with open(fname, "r") as f:
                yield f.read()

    @property
    def sents(self) -> Dict[str, str]:
        if self._sents is not None:
            return self._sents
        nlp = spacy.load("en_core_web_sm")
        sents = {}
        for fname, raw_text in zip(self._text_files, self):
            sents[fname] = [s.replace("\n", "") for s in nlp(raw_text).sents]
        self._sents = sents
        return sents

    @property
    def fulltext(self) -> str:
        all_text = [t.replace("\n", "") for t in self]
        return " ".join(all_text)

    # word - text: str
    #      - pos: str
    #      - docs: list
    #      - sents - [(doc: str, idx: int)]
