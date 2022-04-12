import argparse
from dataclasses import dataclass, field
import logging
from jinja2 import Environment, PackageLoader, select_autoescape
from typing import List, Optional
from keyphrase_vectorizers import KeyphraseCountVectorizer
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span, Token

from keyphrases.textdata import TextData
from keyphrases.jinja_sub_filter import regex_sub


logger = logging.getLogger(__name__)


@dataclass
class Keyphrase:
    text: str
    spans: List[Span] = field(default_factory=list)
    filenames: List[str] = field(default_factory=list)

    @property
    def sentences(self):
        return [s.sent.text.replace("\n", "") for s in self.spans]

    @property
    def count(self):
        return len(self.spans)

    def add_keyphrase(self, span: Span, filename: str):
        if span.text.lower() != self.text.lower():
            raise ValueError(
                f"Invalid span {span.text} to correspond with text {self.text}"
            )
        self.spans.append(span)
        self.filenames.append(filename)


class Keyphrases:
    """Extraction of frequent interesting words from text documents

    Extracted keyphrases are analysed across all input texts, aggregating
    counts. Results can be filtered for totals and text occurances. Rendering
    can be performed on the results, producing an HTML document

    Args:
        fname_glob_pattern (str): glob pattern to find files to read.
            Recursive. Default '*.txt'
        total_freq_thresh (int): Threshold for the frequency to be found
            summed across all documents. Default 2
        cross_doc_freq_thresh (int): Threshold for the number of documents
            the keyphrase must be found, independent on how many times in
            each. Default 2
        pos_pattern (str): part of speach pattern describing the keyphrase
            to be found. Using the pattern gives flexibility for single words
            or complex phrases.
            Example patterns:
                '<N.*>' a single noun
                '<J.*>*<N.*>+' zero or more adjectives followed by one or
                more nouns
            Default is a single noun "<N.*>",

    """

    def __init__(
        self,
        fname_glob_pattern: str = "*.txt",
        total_freq_thresh: int = 2,
        cross_doc_freq_thresh: int = 2,
        pos_pattern: str = "<N.*>",
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

        self.process(pos_pattern)

    @staticmethod
    def run_vectorizer(texts: List[str], pos_pattern: str = "<N.*>"):
        """Run the KeyphraseCountVectorizer on texts to extract keyphrases

        The KeyphraseCountVectorizer analyses all texts, finding a single
        aggregated set of keyphrases but maintaining a count in each text

        Args:
            texts (List[str]): list of strings to be analysed

        """
        vectorizer = KeyphraseCountVectorizer(pos_pattern=pos_pattern, min_df=1)

        doc_keyphrase_matrix = vectorizer.fit_transform(texts).toarray()
        feature_names = vectorizer.get_feature_names_out()

        return feature_names, doc_keyphrase_matrix

    def process(self, pos_pattern: str = "<N.*>"):
        """Process text files to extract interesting frequent words"""
        text_content = list(self._text_data)
        feature_names, keyphrase_matrix = self.run_vectorizer(text_content, pos_pattern)
        self.feature_names = feature_names
        self.keyphrase_matrix = keyphrase_matrix

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
            total_freq_thresh (Optional[int]): threshold for total across all
                text. Default None means use the class value
            cross_doc_freq_thresh (Optional[int]): threshold for occurance in
                each text. Default None means use the class value
        """
        total_freq_thresh = total_freq_thresh or self._total_freq_thresh
        cross_doc_freq_thresh = cross_doc_freq_thresh or self._cross_doc_freq_thresh

        ixs = self.keyphrase_matrix.copy()
        feature_names = self.feature_names

        # Filter according to occurance in each doc
        # Binarise occurance and sum across all docs
        ixs[ixs > 0] = 1
        ixs = ixs.sum(axis=0) >= cross_doc_freq_thresh

        # Zero out any frequencies for words that don't occur in enough
        # documents, regardless of how may times in each
        kp_matrix = self.keyphrase_matrix.sum(axis=0)
        kp_matrix[~ixs] = 0

        ixs = kp_matrix >= total_freq_thresh
        # We want a per-doc matrix so we have filtered words and their
        # occurance in each doc, and a matrix for the filtered words
        # accumulated across all docs
        self._per_doc_keyphrase_matrix = self.keyphrase_matrix.copy()
        self._per_doc_keyphrase_matrix[
            ~ixs[None, :].repeat(self._per_doc_keyphrase_matrix.shape[0], axis=0)
        ] = 0
        self.feature_names_complete = self.feature_names.copy()

        # Keep words whose totals across all docs are above threshold
        kp_matrix = kp_matrix[ixs]
        feature_names = feature_names[ixs]

        self.keyphrase_matrix = kp_matrix
        self.feature_names = feature_names

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
                match_kps = self.feature_names_complete[
                    self._per_doc_keyphrase_matrix[i] > 0
                ]
            patterns = [[{"LOWER": s} for s in kp.split(" ")] for kp in match_kps]
            matcher.add("featurenames", patterns)

            doc = nlp(doc_string)
            matches = matcher(doc, as_spans=True)
            for span in matches:
                kp_text = span.text.lower()
                if kp_text not in matched_sentences:
                    kp = Keyphrase(kp_text)
                    matched_sentences[kp_text] = kp
                else:
                    kp = matched_sentences[kp_text]
                kp.add_keyphrase(span, filename)
                logging.debug(
                    "File: %s | phrase: %s\n\t%s", filename, kp_text, span.sent
                )

        self.matched_sentences = matched_sentences
        return matched_sentences

    def render(self, output_filename: str = "keyphrases.html"):
        """Render results to HTML table

        The keyphrases and associated sentences are rendered to an HTML
        table. HTML is written to file and returned

        Args:
            output_filename (str): filename used to write HTML

        Returns:
            (str): rendered HTML
        """
        env = Environment(
            loader=PackageLoader("keyphrases"), autoescape=select_autoescape
        )
        env.filters["regex_sub"] = regex_sub

        template = env.get_template("render_template.html")

        # Sort keyphrases in decending order of frequency
        keyphrases_sorted = sorted(
            self.matched_sentences.values(), key=lambda v: v.count, reverse=True
        )

        render_html = template.render(
            keyphrase_matches=keyphrases_sorted,
            css_path="keyphrases/templates/style.css",
        )
        with open(output_filename, "w") as f:
            f.write(render_html)

        return render_html

    def match_and_render(self):
        self.match_sentences()
        self.render()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--glob-pattern",
        default="./data/*.txt",
        help="filename glob pattern including directories",
    )
    parser.add_argument(
        "-t",
        "--thresh-total",
        type=int,
        default=2,
        help="threshold for keyphrase frequency summed across all texts",
    )
    parser.add_argument(
        "-c",
        "--thresh-cross-text",
        type=int,
        default=2,
        help="threshold for keyphrase file occurance. must be found once or more in this many files",
    )
    parser.add_argument(
        "-p",
        "--pos-pattern",
        type=str,
        default="<J.*>*<N.*>+",
        help="Part of Speech pattern defining keyphrase. Use `<N.*>` for a single noun",
    )
    args = parser.parse_args()
    keyphrases = Keyphrases(
        args.glob_pattern, args.thresh_total, args.thresh_cross_text, args.pos_pattern
    )
    keyphrases.filter_by_frequency()
    matches = keyphrases.match_sentences()
    _ = keyphrases.render()
