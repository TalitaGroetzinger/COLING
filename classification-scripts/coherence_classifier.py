from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from collections import Counter
from progress.bar import Bar
import json
import numpy as np
import gensim
import pickle
import nltk
from features import get_length_features, get_postags, pos_tags_and_length, get_length_features_context


def check_word_in_context(list_of_wikihow_instances):
    # count all the words in the context and make a dict representation of it
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        source_dict = {}
        word_frequency = Counter()
        match = wikihow_instance['PPDB_Matches']
        context = wikihow_instance['Source_Context_5_Processed']
        ppdb_matches = [elem[0][0] for elem in match]

        # make BOW for tokenized
        tokenized = word_tokenize(context)
        for token in tokenized:
            token = token.lower()
            word_frequency[token] += 1

        for ppdb_match in ppdb_matches:
            if ppdb_match.lower() in dict(word_frequency).keys():
                count_for_match = word_frequency[ppdb_match]
                if count_for_match > 1:
                    source_dict[ppdb_match] = True
                else:
                    source_dict[ppdb_match] = False
        X.append(source_dict)
        Y.append(0)
        # repeat procedure for target
        target_dict = {}
        target_match = wikihow_instance['PPDB_Matches']
        target_context = wikihow_instance['Target_Context_5_Processed']
        target_ppdb_matches = [elem[1][0] for elem in target_match]
        tokenized_target = word_tokenize(target_context)
        word_frequency_target = Counter()
        for token in tokenized_target:
            token = token.lower()
            word_frequency_target[token] += 1

        for ppdb_match in target_ppdb_matches:
            if ppdb_match.lower() in dict(word_frequency).keys():
                count_for_match = word_frequency[ppdb_match]
                if count_for_match > 1:
                    target_dict[ppdb_match] = True
                else:
                    target_dict[ppdb_match] = False
        X.append(target_dict)
        Y.append(1)
    return X, Y


def preprocess_data(train, dev, test):
    with open(train, 'r') as json_in_train:
        train_open = json.load(json_in_train)
    with open(dev, 'r') as json_in_dev:
        dev_open = json.load(json_in_dev)
    with open(test, 'r') as json_in_test:
        test_open = json.load(json_in_test)

    Xtrain, Ytrain = check_word_in_context(train_open)
    Xdev, Ydev = check_word_in_context(dev_open)
    Xtest, Ytest = check_word_in_context(test_open)
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest):
    vec = DictVectorizer()
    # hier moet de dict van Xtrain dus in

    Xtrain_fitted = vec.fit_transform(Xtrain)

    Xdev_fitted = vec.transform(Xdev)

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
    # get_most_informative_features(classifier, count_vec)
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
