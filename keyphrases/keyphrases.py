from dataclasses import dataclass, field
import logging
from typing import List, Optional
from keyphrase_vectorizers import KeyphraseCountVectorizer
from soupsieve import match
from keyphrases.textdata import TextData
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Token

logger = logging.getLogger(__name__)


@dataclass
class Keyphrase:
    text: str
    spans: List[Span] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)

    @property
    def sentences(self):
        return [s.sent.text.replace("\n", "") for s in self.spans]

    def add_keyphrase(self, span: Span, filename: str):
        if span.text != self.text:
            raise ValueError(
                f"Invalid span {span.text} to correspond with text {self.text}"
            )
        self.spans.append(span)
        self.filenames.append(filename)


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

    def filter_by_frequency(
        self,
        total_freq_thresh: Optional[int] = None,
        cross_doc_freq_thresh: Optional[int] = None,
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

        # Zero out any frequencies for words that don't occur in enough
        # documents, regardless of how may times in each
        kp_matrix = self._keyphrase_matrix.sum(axis=0)
        kp_matrix[~ixs] = 0

        ixs = kp_matrix >= total_freq_thresh
        # We want a per-doc matrix so we have filtered words and their
        # occurance in each doc, and a matrix for the filtered words
        # accumulated across all docs
        self._per_doc_keyphrase_matrix = self._keyphrase_matrix.copy()
        self._per_doc_keyphrase_matrix[
            ~ixs[None, :].repeat(self._per_doc_keyphrase_matrix.shape[0], axis=0)
        ] = 0
        self._feature_names_complete = self._feature_names.copy()

        # Keep words whose totals across all docs are above threshold
        kp_matrix = kp_matrix[ixs]
        feature_names = feature_names[ixs]

        self._keyphrase_matrix = kp_matrix
        self._feature_names = feature_names

    def sort(self, reverse: bool = True):
        raise NotImplementedError

    def _filter_by_semantic_similarity(self, n_topk=20):
        # TODO Use KeyBERT to get the semantic similarity of each word
        raise NotImplementedError()

    def match_sentences(
        self, match_keyphrases: Optional[List[str]] = None
    ) -> List[str]:
        """Finds all sentences containing keyphrase strings (words or phrases)

        Uses spaCy matching to find all sentences for each word or phrase
        There is also the `~.matches()` method which does the same matching
        but returns span objects instead which are more flexible and maintain
        all spaCy functionality

        Args:
            match_keyphrases (Optional[List[str]]): a list of keyphrases
                (words or phrases) to find. Default is None meaning we use
                the internal values for `~.feature_names` frequent phrases

        Returns:
            list[str]: a list of strings for each sentence containing a
                keyphrase
        """
        nlp = spacy.load("en_core_web_sm")
        matcher = Matcher(nlp.vocab)

        # Construct match objects
        # Multi-word (phrases) need to be split for each word to
        # patterns = [[{"LOWER": s} for s in kp.split(" ")] for kp in keyphrases]
        # matcher.add(patterns)

        matched_sentences = {}
        for i, (filename, doc_string) in enumerate(
            zip(self._text_data.filenames, self._text_data)
        ):
            # Construct match objects
            # Multi-word (phrases) need to be split for each word to
            if match_keyphrases is not None:
                match_kps = match_keyphrases
            else:
                # use internal frequent values for this doc
                match_kps = self._feature_names_complete[
                    self._per_doc_keyphrase_matrix[i] > 0
                ]
            patterns = [[{"LOWER": s} for s in kp.split(" ")] for kp in match_kps]
            matcher.add("featurenames", patterns)

            doc = nlp(doc_string)
            matches = matcher(doc, as_spans=True)
            for span in matches:
                kp_text = span.text
                if kp_text not in matched_sentences:
                    kp = Keyphrase(kp_text)
                    matched_sentences[kp_text] = kp
                else:
                    kp = matched_sentences[kp_text]
                kp.add_keyphrase(span, filename)
                logging.debug("File: %s | phrase: %s\n\t%s", filename, kp, span.sent)

        self.matched_sentences = matched_sentences
        return matched_sentences

        # file_occurances = {}
        # sentences = {}
        # for filename, doc_string in zip(self._text_data.filenames, self._text_data):
        #     doc = nlp(doc_string)
        #     matches = matcher(doc, as_spans=True)
        #     for span in matches:
        #         kp = span.text
        #         file_occurances.get(kp, set()).add(filename)
        #         sentences.get(kp, []).append(span.sent)
        #         logging.debug("File: %s | phrase: %s\n\t%s", filename, kp, span.sent)

        """
        span
        sent
        filename
        

        for s in span
            span
            all_filenames = span_filenames[span]
            all_sents = 
        
        """
