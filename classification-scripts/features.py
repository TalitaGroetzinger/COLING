import gensim
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import nltk
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from progress.bar import Bar

path_to_markers = '../data/discourse_markers.pickle'
with open(path_to_markers, 'rb') as pickle_in:
    markers = pickle.load(pickle_in)


def regex_tokeniser(x):
    tokenizer = RegexpTokenizer('[^ ]+')
    return tokenizer.tokenize(x)


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


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def get_length_features(document, thresshold=15):
    length_doc = len(document)
    if length_doc > 15:
        length_type = "long"
    else:
        length_type = "short"
    return [word + '_' + length_type for word in document]


def get_length_features_context(document, thresshold=150):
    length_doc = len(document)
    if length_doc > 170:
        length_type = "long"
    else:
        length_type = "short"
    return [word + '_' + length_type for word in document]


def get_postags(tokens):
    '''
        Returns part-of-speech tags
    '''
    # use first_tokens for context with __REV__
    first_tokens = word_tokenize(tokens.replace("__REV___", ""))
    return [token + "_POS-"+tag for token, tag in nltk.pos_tag(first_tokens)]


def tokenize(tokens):
    """
        Can be used to not use POS tags in the context. 
    """
    tokens = word_tokenize(tokens.replace("__REV___", ""))
    return tokens


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


def check_discourse_matches(tokens, markers):
    """
        Input: tokenized sent or document from wikihow_instance
    """
    total = 0
    unigram_matches = 0
    bigram_matches = 0
    trigram_matches = 0
    fourgram_matches = 0
    fivegram_matches = 0
    print("-----")
    # Later zal dit meteen het document zijn/de tokens.
    #tokens = word_tokenize(tokens)
    #tokens = regex_tokeniser(tokens)
    for token in tokens:
        if token in markers.keys():
            if 'fivegrams' in markers[token].keys():
                fivegrams = [[tokens[i], tokens[i+1], tokens[i+2],
                              tokens[i+3], tokens[i+4]] for i in range(len(tokens)-5)]
                for fivegram in fivegrams:
                    if fivegram in markers[token]['fivegrams']:
                        fivegram_matches += 1
                        total += 1
                        # print(fivegram_matches, '#',
                        #      markers[token]['fivegrams'])
                        # print("\n")

            if 'fourgrams' in markers[token].keys():
                fourgrams = [[tokens[i], tokens[i+1], tokens[i+2],
                              tokens[i+3]] for i in range(len(tokens)-4)]
                for fourgram in fourgrams:
                    if fourgram in markers[token]['fourgrams']:
                        fourgram_matches += 1
                        total += 1
                        # print(fourgram, '#', markers[token]['fourgrams'])
                        # print("\n")

            if 'trigrams' in markers[token].keys():
                trigrams = [[tokens[i], tokens[i+1], tokens[i+2]]
                            for i in range(len(tokens)-3)]
                for trigram in trigrams:
                    if trigram in markers[token]['trigrams']:
                        trigram_matches += 1
                        total += 1
                        # print(trigram, '#', markers[token]['trigrams'])
                        # print("\n")

            if 'bigrams' in markers[token].keys():
                bigrams = [[tokens[i], tokens[i+1]]
                           for i in range(len(tokens)-2)]
                for bigram in bigrams:
                    if bigram in markers[token]['bigrams']:
                        bigram_matches += 1
                        total += 1
                        #print(bigram, markers[token]['bigrams'])
                        # print("\n")
            if 'unigrams' in markers[token].keys():
                # print(token, '#', markers[token]['unigrams'])
                unigram_matches += 1
                total += 1
    return {"score": unigram_matches + bigram_matches + trigram_matches + fourgram_matches + fivegram_matches}


class DiscourseFeatures(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def _get_features(self, doc):
        discourse_score = check_discourse_matches(doc, markers)
        return discourse_score

    def transform(self, raw_documents):
        features = []
        bar = Bar('Processing ', max=len(raw_documents))
        for doc in raw_documents:
            bar.next()
            features.append(self._get_features(doc))
        return features

        # return [self._get_features(doc) for doc in raw_documents]


discourse_vec = Pipeline(
    [
        ('feat', DiscourseFeatures()), ('vec', DictVectorizer())
    ]
)


class LexicalComplexity(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def _get_features(self, doc):
        lexical_score = type_token_ratio(doc)
        return lexical_score

    def transform(self, raw_documents):
        return [self._get_features(doc) for doc in raw_documents]


def type_token_ratio(document, regex=False):
    """
        Input: tokenize document 
        Returns: the type-token-ratio for a document.
    """
    if regex:
        all_tokens = regex_tokeniser(document)
    else:
        print("do not use regex")
        all_tokens = word_tokenize(document)
    num_of_tokens = len(all_tokens)
    unique_tokens = list(set(all_tokens))
    num_of_unique_tokens = len(unique_tokens)
    return {"Type-token-ratio": num_of_unique_tokens/num_of_tokens}


lexical_complexity_vec = Pipeline(
    [
        ('feat', LexicalComplexity()), ('vec', DictVectorizer())
    ]
)
