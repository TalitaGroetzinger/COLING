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
from features import get_length_features, get_postags, tokenize, pos_tags_and_length, get_length_features_context, coherence_vec, discourse_vec


def mark_cases(context, matches, source=True):
    if source:
        match = [elem[0][0].lower() for elem in matches]
    else:
        match = [elem[1][0].lower() for elem in matches]

    document = []
    for sent in context:
        tokenized = word_tokenize(sent)
        tokenized = [word[0] for word in nltk.pos_tag(tokenized)]
        for word in tokenized:
            document.append(word)

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


# let op dat er bij deze functie ook gebruik wordt gemaakt van de matches
# voor context
def get_docs_labels(list_of_wikihow_instances, context=False, diff_noun_file=True):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        if context:
            print("use context ... ")
            source_context_first = wikihow_instance['Source_Context_5_Processed']
            source_context = word_tokenize(source_context_first)
            target_context_first = wikihow_instance['Target_Context_5_Processed']
            target_context = word_tokenize(target_context_first)
            #matches = wikihow_instance['PPDB_Matches']
            X.append(source_context)
            Y.append(0)
            X.append(target_context)
            Y.append(1)
        else:
            print("do not use context")
            if diff_noun_file:
                print("use different noun modifications")
                source_tokenized = [pair[0]
                                    for pair in wikihow_instance['Source_tagged']]
                target_tokenized = [pair[0]
                                    for pair in wikihow_instance['Target_Tagged']]
            else:
                source_tokenized = [pair[0]
                                    for pair in wikihow_instance['Source_Line_Tagged']]
                target_tokenized = [pair[0]
                                    for pair in wikihow_instance['Target_Line_Tagged']]
            X.append(source_tokenized)
            Y.append(0)
            X.append(target_tokenized)
            Y.append(1)
    return X, Y


def preprocess_data(train_diff, dev_diff, test_diff, train_same, dev_same, test_same, use_context=False):
    with open(train_diff, 'r') as json_in_train:
        train_open_diff = json.load(json_in_train)
    with open(dev_diff, 'r') as json_in_dev:
        dev_open_diff = json.load(json_in_dev)
    with open(test_diff, 'r') as json_in_test:
        test_open_diff = json.load(json_in_test)

    with open(train_same, 'r') as json_in_train_same:
        train_open_same = json.load(json_in_train_same)

    with open(dev_same, 'r') as json_in_dev_same:
        dev_open_same = json.load(json_in_dev_same)

    with open(test_same, 'r') as json_in_test_same:
        test_open_same = json.load(json_in_test_same)

    # list_of_wikihow_instances, context=False, use_matches=False, diff_noun_file=True
    if use_context:
        Xtrain_diff, Ytrain_diff = get_docs_labels(
            train_open_diff, context=True)
        Xdev_diff, Ydev_diff = get_docs_labels(
            dev_open_diff, context=True)
        Xtest_diff, Ytest_diff = get_docs_labels(
            test_open_diff, context=True)

        Xtrain_same, Ytrain_same = get_docs_labels(
            train_open_same, context=True)
        Xdev_same, Ydev_same = get_docs_labels(
            dev_open_same, context=True)
        Xtest_same, Ytest_same = get_docs_labels(
            test_open_same, context=True)
    else:
        Xtrain_diff, Ytrain_diff = get_docs_labels(
            train_open_diff, context=False, diff_noun_file=True)
        Xdev_diff, Ydev_diff = get_docs_labels(
            dev_open_diff, context=False, diff_noun_file=True)
        Xtest_diff, Ytest_diff = get_docs_labels(
            test_open_diff, context=False, diff_noun_file=True)

        Xtrain_same, Ytrain_same = get_docs_labels(
            train_open_same, context=False, diff_noun_file=False)
        Xdev_same, Ydev_same = get_docs_labels(
            dev_open_same, context=False, diff_noun_file=False)
        Xtest_same, Ytest_same = get_docs_labels(
            test_open_same, context=False, diff_noun_file=False)

    Xtrain = Xtrain_diff + Xtrain_same
    Ytrain = Ytrain_diff + Ytrain_same
    Xdev = Xdev_diff + Xdev_same
    Ydev = Ydev_diff + Ydev_same
    Xtest = Xtest_diff + Xtest_same
    Ytest = Ytest_diff + Ytest_same

    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def join_data(x):
    return ' '.join(x)


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest):
    # hier moet de dict van Xtrain dus in
    # fit to countvec
    count_vec = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 2),
                                token_pattern='[^ ]+', preprocessor=join_data)
    """
    Xtrain_fitted = count_vec.fit_transform(Xtrain)
    Xdev_fitted = count_vec.transform(Xdev)
    """
    print("fit data ... ")
    vec = FeatureUnion(
        [
            ('feat', discourse_vec), ('vec', count_vec)
        ]
    )

    Xtrain_fitted = vec.fit_transform(Xtrain)
    Xdev_fitted = vec.transform(Xdev)
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
    get_most_informative_features(classifier, vec)
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
    train_diff, dev_diff, test_diff = get_paths(True)
    train_same, dev_same, test_same = get_paths(False)

    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = preprocess_data(
        train_diff, dev_diff, test_diff, train_same, dev_same, test_same, use_context=True)

    list_of_good_predictions, list_of_bad_predictions = train_classifier(
        Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest)


main()
