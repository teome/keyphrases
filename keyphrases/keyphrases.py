from typing import List
from keyphrase_vectorizers import KeyphraseCountVectorizer
from keyphrases.textdata import TextData


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

        self._keyphrase_matrix_filtered = kp_matrix
        self._feature_names_filtered = feature_names

        # TODO sort

    # def match(self, words: List[str]) -> TODO
