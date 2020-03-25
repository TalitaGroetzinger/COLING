from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from progress.bar import Bar
import json
import numpy as np
import gensim
import pickle
import nltk
from features import get_length_features, get_postags, pos_tags_and_length, get_length_features_context, coherence_vec


def mark_cases(context, matches, source=True):
    if source:
        match = [elem[0][0].lower() for elem in matches]
    else:
        match = [elem[1][0].lower() for elem in matches]

    document = word_tokenize(context)

    final_rep = []
    for word in document:
        if word.lower() in match:
            word = word + "__REV__"
            final_rep.append(word)
        else:
            final_rep.append(word)
    return final_rep


def get_most_informative_features(classifier, vec, top_features=10):
    print("------------------------------------")
    print("---- Most informative features -----")
    print("------------------------------------")
    neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()
    pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()
    print("Source: ", np.take(
        vec.get_feature_names(), neg_class_prob_sorted[:top_features]))
    print("Target: ", np.take(
        vec.get_feature_names(), pos_class_prob_sorted[:top_features]))


def get_docs_labels_context(list_of_wikihow_instances):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        source_context = wikihow_instance['Source_Context_5_Processed']
        target_context = wikihow_instance['Target_Context_5_Processed']
        matches = wikihow_instance['PPDB_Matches']

        new_source = mark_cases(source_context, matches, source=True)
        new_target = mark_cases(target_context, matches, source=False)

        X.append(new_source)
        Y.append(0)
        X.append(new_target)
        Y.append(1)
    return X, Y


def preprocess_data(train, dev, test):
    with open(train, 'r') as json_in_train:
        train_open = json.load(json_in_train)
    with open(dev, 'r') as json_in_dev:
        dev_open = json.load(json_in_dev)
    with open(test, 'r') as json_in_test:
        test_open = json.load(json_in_test)

    Xtrain, Ytrain = get_docs_labels_context(train_open)
    Xdev, Ydev = get_docs_labels_context(dev_open)
    Xtest, Ytest = get_docs_labels_context(test_open)
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest):
    # hier moet de dict van Xtrain dus in
    # fit to countvec
    """
    count_vec = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 1),
                                tokenizer=pos_tags_and_length, preprocessor=word_tokenize)
    Xtrain_fitted = count_vec.fit_transform(Xtrain)
    Xdev_fitted = count_vec.transform(Xdev)
    """

    Xtrain_fitted = coherence_vec.fit_transform(Xtrain)
    Xdev_fitted = coherence_vec.transform(Xdev)
    # ------------------------------------------------

    # classification
    classifier = MultinomialNB()
    classifier.fit(Xtrain_fitted, Ytrain)

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
    print("Accuracy: {0}".format(accuracy))
    get_most_informative_features(classifier, count_vec)
    return list_of_good_predictions, list_of_bad_predictions


def get_paths(different_nouns=True):
    if different_nouns:
        path_to_train = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'
        path_to_dev = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-DEV-5-new.JSON'
        path_to_test = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TEST-5-new.JSON'
    else:
        path_to_train = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'
        path_to_dev = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-DEV-5-new.JSON'
        path_to_test = './same-noun-modifications/SAME-NOUN-MODIFICATIONS-TEST-5-new.JSON'
    return path_to_train, path_to_dev, path_to_test


def main():
    # get different nouns
    train, dev, test = get_paths(True)

    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = preprocess_data(
        train, dev, test)

    list_of_good_predictions, list_of_bad_predictions = train_classifier(
        Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest)


main()
