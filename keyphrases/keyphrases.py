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

        # TODO sort

    # def cross_doc_frequency(self, cross_doc_freq_thresh: int = None):
    #     """Word frequencies across docs/texts

    #     Process the vectorizer's keyphrase_matrix across all docs to count presence
    #     in each. Thresholding is done w.r.t. number of docs containing the word any
    #     number of times (anything > 0). Once thresholded, we count word frequencies.
    #     This allows finding interesting common words between docs/texts without
    #     letting any word occuring very frequently in a single doc overpowering
    #     results

    #     Args:
    #       cross_doc_freq_thresh (int): threshold for number of docs the word must
    #           occur in for it to be included, otherwise it's removed from the
    #           results. Default None means we use the class member value
    #           `._cross_doc_freq_thresh`

    #     Returns:
    #       dict[str, int]: dict mapping between words and their counts
    #     """
    #     if cross_doc_freq_thresh is None:
    #         cross_doc_freq_thresh = self._cross_doc_freq_thresh

    #     # Our keyphrase_matrix has shape [n_strings, n_words] i.e. a n_word array
    #     # for each of the strings/documents in our original array.
    #     # Create binary arrays for occurance any number
    #     # of times above 0 in each doc. This is used to calculate
    #     # frequency/presence of each word across docs rather than the total
    #     cross_doc_keyphrase_matrix = self._doc_keyphrase_matrix.copy()
    #     cross_doc_keyphrase_matrix[cross_doc_keyphrase_matrix > 1] = 1
    #     cross_doc_keyphrase_matrix = cross_doc_keyphrase_matrix.sum(axis=0)

    #     # Create indices to extract results that occur at least `cross_doc_freq_thresh`
    #     # documents. This ignores how many times in each document above 1
    #     above_thresh_ixs = cross_doc_keyphrase_matrix >= cross_doc_freq_thresh
    #     above_thresh_ixs = above_thresh_ixs.reshape(1, -1).repeat(
    #         6, axis=0
    #     )  # .astype(np.int)

    #     # TODO multiply by ints mask rather than index with True/False. This will keep
    #     # the shape and allow merging with the result of filtering per=doc. So change this
    #     # such that we keep shape, filter, merge, then treat as a whole. too complicated otherwise
    #     # Lose the shape so have to reshape to return to get the document dimension back
    #     thresholded_doc_keyphrase_matrix = self._doc_keyphrase_matrix[
    #         above_thresh_ixs
    #     ].reshape(doc_keyphrase_matrix.shape[0], -1)
    #     # We just have the single matrix for the words themselves
    #     thresholded_feature_names = self._feature_names[above_thresh_ixs[0]]

    #     # We now have the desired thresholded results and can look at counts in all docs
    #     # and in individual docs
    #     total_doc_keyphrase_matrix = thresholded_doc_keyphrase_matrix.sum(axis=0)
    #     self._cross_doc_totals_dict = {
    #         k: v for k, v in zip(thresholded_feature_names, total_doc_keyphrase_matrix)
    #     }

    #     print(sorted(totals_dict.items(), reverse=True, key=lambda kv: kv[1]))

    # def per_doc_frequency(self, per_doc_freq_thresh: int = None):
    #     """Calculates word frequences independently in each doc/string

    #     Thresholding is done on a per-doc level. Words occuring above the threshold
    #     in any doc/string are then allowed into the results

    #     This is independent of the cross-doc processing and complementary. It allows
    #     finding interesting words specific to each doc which may or may not be
    #     thematic across docs

    #     Args:
    #       per_doc_freq_thresh (int): threshold for number times a word must occur
    #           in its doc/string for it to be included in the results. Default None
    #           means we use the class `._per_doc_freq_thresh`

    #     Returns:
    #       dict[str, dict[str, int]]: dict mapping between filenames and
    #           a dict mapping between words and their counts
    #     """
    #     if per_doc_freq_thresh is None:
    #         per_doc_freq_thresh = self._per_doc_freq_thresh

    #     all_docs_dict = {}
    #     for filename, kp_matrix in zip(self._filenames, self._doc_keyphrase_matrix):
    #         above_thresh_ixs = kp_matrix >= per_doc_freq_thresh
    #         single_doc_dict = {
    #             k: v
    #             for k, v in zip(
    #                 feature_names[above_thresh_ixs], kp_matrix[above_thresh_ixs]
    #             )
    #         }
    #         all_docs_dict[file_name] = single_doc_dict
    #         print(sorted(single_doc_dict.items(), reverse=True, key=lambda kv: kv[1]))

    # def match(self, words: List[str]) -> TODO
