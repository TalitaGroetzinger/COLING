from classifier import get_data
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from classifier import get_data, get_XY, split_data_by_article, load_json_file
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle
import gensim
from features import MeanEmbeddingVectorizer


def tokenize(X):
    collection = []
    for sents in X:
        tokenized_sent = []
        sents = sents.split()
        for word in sents:
            tokenized_sent.append(word)
        collection.append(tokenized_sent)
    return collection


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, ngram_range_value=(1, 1), word2vec=True):
    """
        Slightly modified from Irshad.
    """
    # vectorize data
    print("Vectorize the data ...")
    if not word2vec:
        vec = TfidfVectorizer(max_features=None, lowercase=False,
                              ngram_range=ngram_range_value, stop_words=None, token_pattern='[^ ]+')
    else:
        model = gensim.models.Word2Vec(
            Xtrain, size=100, window=5, min_count=1, workers=2)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        vec = MeanEmbeddingVectorizer(w2v)

    classifier = Pipeline(
        [('vec', vec), ('cls', LinearSVC())])

    classifier.fit(Xtrain, Ytrain)
    print("Finished training ..")
    YpredDev = classifier.predict(tokenize(Xdev))
    accuracy = accuracy_score(YpredDev, Ydev)
    report = classification_report(YpredDev, Ydev)
    print(accuracy)
    print(report)


def main():
    # read json file
    list_of_wikihow_instances = load_json_file(False)
    # get splits by article
    train_set_files = pickle.load(
        open("./classification-data/train_set_files.pickle", "rb"))
    dev_set_files = pickle.load(
        open("./classification-data/dev_set_files.pickle", "rb"))
    test_set_files = pickle.load(
        open("./classification-data/test_set_files.pickle", "rb"))

    # split further into X and Y
    Xtrain, Ytrain, Xdev, Ydev = get_XY(
        list_of_wikihow_instances, train_set_files, dev_set_files, test_set_files, use_test_set=False)
    train_classifier(Xtrain, Ytrain, Xdev, Ydev)


main()
