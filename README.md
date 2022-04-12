# Frequently Interesting Words

Simple module which analyses a set of text files to find keyphrases. The keyphrases are specified using a Parts of Speech (POS) pattern which allows for a lot of flexibility, from just looking for key words (e.g. nouns) to more complex phrases composed of an adjective followed by a noun.

The keyphrases can be filtered in two independent ways using thresholds:

- On the total frequency of the phrase occurance
- On the number of files in which the phrase occurs, independent on the number of times in each

A simple approach of just counting each word and sorting by the highest frequency is not taken in favour of a far more flexible and extensible approach using POS and CountVectorizers.

## Results rendering

The results can be rendered in HTML form as a table showing for each each word/phrase its:

- Total occurance count across all files
- Sentence containing the word/phrase
- The file the phrase and sentence was found in

## Libraries

- Under the hood [spacy](https://spacy.io) is used for a lot of the text processing.
- Frequency analysis, POS parsing and processing to find the keyphrases themselves is handled by the [KeyphraseVectorizers](https://github.com/TimSchopf/KeyphraseVectorizers) library.

It's not currently implemented but these libraries easily allow for extending the approach to use more advance NLP, e.g. language models to calculate the semantic similarity of each keyphrase to the documents themselves, and to find semantically similar sentences.

## Installation

The easiest installation is via pip. As is standard, I would recommend installing into a viatual environment either via `venv`

```python
python3 -m venv keyphrases-env
source keyphrases-env/bin/activate
```

Or using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Then use the `requirements.txt` provided

```python
python3 -m pip install -r requirements.txt
```

## Example usage

The simplest use is via the `keyphrases` module itself which includes the `argparse` library for command line usage

```python
python -m keyphrases.keyphrases --help
usage: keyphrases.py [-h] [-g GLOB_PATTERN] [-t THRESH_TOTAL] [-c THRESH_CROSS_TEXT] [-p POS_PATTERN]

optional arguments:
  -h, --help            show this help message and exit
  -g GLOB_PATTERN, --glob-pattern GLOB_PATTERN
                        filename glob pattern including directories
  -t THRESH_TOTAL, --thresh-total THRESH_TOTAL
                        threshold for keyphrase frequency summed across all texts
  -c THRESH_CROSS_TEXT, --thresh-cross-text THRESH_CROSS_TEXT
                        threshold for keyphrase file occurance. must be found once or more in this many files
  -p POS_PATTERN, --pos-pattern POS_PATTERN
                        Part of Speech pattern defining keyphrase. Use `<N.*>` for a single noun
```

A typical use for a directory `data` containing text files, a keyphrase defined by zero or more adjectives and one or more nouns, filtering by at least a total of 3 occurances and occuring in 2 files would look like

```python
keyphrases = Keyphrases('./data/*.txt', 3, 2, '<J.*>*<N.*>+')
keyphrases.filter_by_frequency()
matches = keyphrases.match_sentences()
keyphrases.render('render_output.html')
```

Or to find only the most common interesting words, we can achive this using the noun POS

```python
keyphrases = Keyphrases('./data/*.txt', 1, 1, '<N.*>')
keyphrases.filter_by_frequency()
matches = keyphrases.match_sentences()
keyphrases.render('render_output.html')
```
