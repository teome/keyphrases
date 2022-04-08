from typing import List, Optional
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keyphrases.textdata import TextData

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Token


class Keyphrases:
    """Extraction of frequent interesting words from text documents"""

    def __init__(
        self,
        fname_glob_pattern: str = "*.txt",
        total_freq_thresh: int = 2,
        cross_doc_freq_thresh: int = 2,
    ):
        self._text_data = TextData(fname_glob_pattern)
        if len(self._text_data) < 1:
            raise RuntimeError(
                f"Failed to find any text documents using pattern {fname_glob_pattern}"
            )
        self._filenames = self._text_data.filenames

        if total_freq_thresh < 1:
            raise ValueError(
                f"`total_freq_thresh` must be > 1 but got {total_freq_thresh}"
            )
        if cross_doc_freq_thresh < 1:
            raise ValueError(
                f"`cross_doc_freq_thresh` must be > 1 but got {cross_doc_freq_thresh}"
            )
        self._total_freq_thresh = total_freq_thresh
        self._cross_doc_freq_thresh = cross_doc_freq_thresh

        self.process()

    @staticmethod
    def run_vectorizer(texts: List[str], pos_pattern: str = "<N.*>"):
        vectorizer = KeyphraseCountVectorizer(pos_pattern=pos_pattern, min_df=1)

        doc_keyphrase_matrix = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out()

        return feature_names, doc_keyphrase_matrix

    def process(self, pos_pattern: str = "<N.*>"):
        """Process text files to extract interesting frequent words"""
        # This function can't deal with generators annoyingly
        text_content = list(self._text_data)
        feature_names, keyphrase_matrix = self.run_vectorizer(text_content, pos_pattern)
        self._feature_names = feature_names
        self._keyphrase_matrix = keyphrase_matrix

    def _filter_by_frequency(
        self, total_freq_thresh: int = None, cross_doc_freq_thresh: int = None
    ):
        """Filter common word results by frequency thresholds

        Seperate thresholds are used for cross-doc frequency and per-doc frequency.
        In the case of cross-doc frequency we count only that the word occurs at
        least once in each doc and ignore occurances above this. The threshold
        allows to see how common a word or theme is among all docs. Without this, a
        word could appear in just one doc many times and be counted as the most
        frequent word.

        Conversely, total frequency threshold uses the total count across all
        documents

        Args:
          TODO

        Returns:
          TODO
        """
        total_freq_thresh = total_freq_thresh or self._total_freq_thresh
        cross_doc_freq_thresh = cross_doc_freq_thresh or self._cross_doc_freq_thresh

        ixs = self._keyphrase_matrix.copy()
        feature_names = self._feature_names

        # Filter according to occurance in each doc
        # Binarise occurance and sum across all docs
        ixs[ixs > 0] = 1
        ixs = ixs.sum(axis=0) >= cross_doc_freq_thresh

        # Use this as indices into the original keyphrase matrix, zero then
        # check totals
        # kp_matrix = self._keyphrase_matrix.copy()
        # kp_matrix[~ixs[None, :].repeat(kp_matrix.shape[0], axis=0)] = 0

        # ixs = kp_matrix.sum(axis=0) >= total_freq_thresh
        # ixs = ixs[None, :].repeat(kp_matrix.shape[0], axis=0)
        # kp_matrix = kp_matrix[ixs]
        # feature_names = feature_names[ixs]

        kp_matrix = self._keyphrase_matrix.sum(axis=0)
        kp_matrix[~ixs] = 0

        ixs = kp_matrix >= total_freq_thresh
        kp_matrix = kp_matrix[ixs]
        feature_names = feature_names[ixs]

        # TODO: Should these be set to new variables or overwrite if we don't
        # need the originals
        self._keyphrase_matrix = kp_matrix
        self._feature_names = feature_names

        # TODO sort

    def _filter_by_semantic_similarity(self, n_topk=20):
        raise NotImplementedError()

    # def matches(self, words: List[str]) -> List[Token]

    def match_sentences(self, keyphrases: List[str]) -> List[str]:
        """Finds all sentences containing keyphrase strings (words or phrases)

        Uses spaCy matching to find all sentences for each word or phrase
        There is also the `~.matches()` method which does the same matching
        but returns span objects instead which are more flexible and maintain
        all spaCy functionality

        Args:
            keyphrases (List[str]): a list of keyphrases (words or phrases)
                to find

        Returns:
            list[str]: a list of strings for each sentence containing a
                keyphrase
        """
        if not keyphrases:
            raise ValueError("`keyphrases` cannot be an empty list")

        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)

        # Construct match objects
        patterns = [[{"LOWER": s} for s in kp.split(" ")] for kp in keyphrases]
        matcher.add(patterns)

        file_occurances = {}
        sentences = {}
        for filename, doc_string in zip(self._text_data.filenames, self._text_data):
            doc = nlp(doc_string)
            matches = matcher(doc, as_spans=True)
            for span in matches:
                kp = span.text
                file_occurances.get(kp, set()).add(filename)
                sentences.get(kp, []).append(span.sent)
