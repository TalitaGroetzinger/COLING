import pickle

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.tokenize import word_tokenize
from features import *
from sklearn.naive_bayes import MultinomialNB


def train_data(train, dev, test):
    count_vec = CountVectorizer(max_features=None, lowercase=False,
                                ngram_range=(1, 2), tokenizer=word_tokenize)

    vec1 = Pipeline([
        ('selector', ItemSelector(key='X_Line')
         ), ('count_vec', count_vec),
    ])

    vec2 = Pipeline([
        ('selector', ItemSelector(key='X_Line_Marked')
         ), ('coherence_vec', coherence_vec),
    ])

    vec = FeatureUnion([('vec1', vec1), ('vec2', vec2)])

    print("fit data ")
    #vec = vec1
    Xtrain_fitted = vec.fit_transform(train)
    normalize(Xtrain_fitted)

    Xdev_fitted = vec.transform(dev)
    normalize(Xdev_fitted)
    # ------------------------------------------------
    # normalize(Xtrain_fitted)
    # normalize(Xdev_fitted)
    # classification
    print("classify")
    classifier = MultinomialNB()
    classifier.fit(Xtrain_fitted, train["Y"])

    print("Finished training ..")

    YpredictDev = classifier.predict_proba(Xdev_fitted)[:, 1]
    positive = 0
    negative = 0
    list_of_good_predictions = []
    list_of_bad_predictions = []
    for i, (source_prediction, target_prediction) in enumerate(zip(YpredictDev[::2], YpredictDev[1::2])):
        if source_prediction < target_prediction:
            positive += 1
            list_of_good_predictions.append(i)
        else:
            negative += 1
            list_of_bad_predictions.append(i)
    accuracy = (positive/(positive+negative))
    print(positive+negative)
    print("Accuracy: {0}".format(accuracy))
    return list_of_good_predictions, list_of_bad_predictions


def main():

    with open("test_marked_spec_line.pickle", "rb") as pickle_test:
        test_marked = pickle.load(pickle_test)

    with open("train_marked_spec_line.pickle", "rb") as pickle_train:
        train_marked = pickle.load(pickle_train)

    with open("dev_marked_spec_line.pickle", "rb") as pickle_dev:
        dev_marked = pickle.load(pickle_dev)

    train_data(train_marked, dev_marked, test_marked)


main()
