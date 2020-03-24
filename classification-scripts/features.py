import gensim
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer


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
