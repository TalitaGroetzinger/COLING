import gensim
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.pipeline import Pipeline
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


def check_word_in_context(context, matches):
    # count all the words in the context and make a dict representation of it
    word_frequency = Counter()
    ppdb_matches = [elem[0][0] for elem in matches]

    tokenized = word_tokenize(context)
    for token in tokenized:
        token = token.lower()
        word_frequency[token] += 1
    d = {}
    for ppdb_match in ppdb_matches:
        if ppdb_match.lower() in dict(word_frequency).keys():
            count_for_match = word_frequency[ppdb_match]
            if count_for_match > 1:
                d[ppdb_match] = True
            else:
                d[ppdb_match] = False
    return d


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(word2vec))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def get_length_features(document, thresshold=15):
    length_doc = len(document)
    if length_doc > 15:
        length_type = "long"
    else:
        length_type = "short"
    return [word + '_' + length_type for word in document]


def get_length_features_context(document, thresshold=150):
    length_doc = len(document)
    if length_doc > 150:
        length_type = "long"
    else:
        length_type = "short"
    return [word + '_' + length_type for word in document]


def get_postags(tokens):
    '''
        Returns part-of-speech tags
    '''
    return [token + "_POS-"+tag for token, tag in nltk.pos_tag(tokens)]


def pos_tags_and_length(document, thresshold=150):
    length_doc = len(document)
    if length_doc > 150:
        length_type = "long"
    else:
        length_type = "short"
    return [word + '_' + tag + '_' + length_type for word, tag in nltk.pos_tag(document)]


lrec_vec = CountVectorizer(max_features=None, lowercase=False,
                           ngram_range=(1, 2), stop_words=None, token_pattern='[^ ]+')


def compute_coherence(doc):
    """
        Input: X formatted with __REV__
    """

    freq = Counter()
    d = {}
    print("DOC")
    for word in doc:
        if '__REV__' in word:
            freq[word.lower()] += 1
    bow = dict(freq)
    coherence_score = 0
    for key, _ in bow.items():
        if bow[key] > 1:
            coherence_score += 1
        else:
            coherence_score += 0
    score = coherence_score / len(bow)
    d["score"] = score
    return d


class CoherenceFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def _get_features(self, doc):
        try:
            coherence = compute_coherence(doc)
        # there is one case for which we can not compute the coherence due to tokenisation issues (T.V cannot become t.v.)
        # since it's only once case, I will just provide the score here.
        except ZeroDivisionError:
            coherence = {"score": 1.0}
        return coherence

    def transform(self, raw_documents):
        return [self._get_features(doc) for doc in raw_documents]


coherence_vec = Pipeline(
    [
        ('feat', CoherenceFeatures()), ('vec', DictVectorizer())
    ]
)


class PreprocessFeatures(object):

    def fit(self, X, y=None):
        return self

    def untokenize(self, document):
        return ' '.join(document)

    def transform(self, X):
        return [self.untokenize(document) for document in X]
