from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from collections import Counter
import json
import numpy as np
import gensim
import pickle


def load_json(different_noun_modifications=True):
    if different_noun_modifications:
        #path = './classification-data/DIFF-NOUN-MODIFICATIONS.json'
        path = './classification-data/DIFF-NOUN-MODIFICATIONS-CONTEXT.json'
        print("Using file with DIFF-NOUN-MODIFICATIONS: ", path)
        with open(path, 'r') as json_in:
            list_of_wikihow_instances = json.load(json_in)

    else:
        path = './classification-data/SAME-NOUN-MODIFICATIONS.json'
        print("Using file with SAME-NOUN-MODIFICATIONS: ", path)
        with open(path, 'r') as json_in:
            list_of_wikihow_instances = json.load(json_in)
    return list_of_wikihow_instances


def get_XY(list_of_wikihow_instances, use_test_set=False, diff_noun_file=True):
    Xdev = []
    Ydev = []
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    if diff_noun_file:
        print("USE DIFF NOUN FILE ...")
    else:
        print("USE SAME NOUN FILE .... ")
    for wikihow_instance in list_of_wikihow_instances:
        # this  if-else is necessary because the two json files have different keys :(
        if diff_noun_file:
            source_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Source_tagged']])
            target_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Target_Tagged']])
        else:
            source_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Source_Line_Tagged']])
            target_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Target_Line_Tagged']])
        if wikihow_instance['Loc_in_splits'] == 'TRAIN':
            Xtrain.append(source_untokenized)
            Ytrain.append(0)
            Xtrain.append(target_untokenized)
            Ytrain.append(1)
        elif wikihow_instance['Loc_in_splits'] == 'DEV':
            Xdev.append(source_untokenized)
            Ydev.append(0)
            Xdev.append(target_untokenized)
            Ydev.append(1)
        else:
            Xtest.append(source_untokenized)
            Ytest.append(0)
            Xtest.append(target_untokenized)
            Ytest.append(1)
    print("Train cases:", len(Xtrain))
    print("Dev cases: ", len(Xdev))
    print("test cases: ", len(Xtest))
    if use_test_set:
        return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest
    else:
        return Xtrain, Ytrain, Xdev, Ydev


def get_docs_labels(list_of_wikihow_instances, diff_noun_file=True):
    X = []
    Y = []
    if diff_noun_file:
        print("use DIFF-NOUNS")
    else:
        print("use SAME-NOUNS")
    for wikihow_instance in list_of_wikihow_instances:
        if diff_noun_file:
            source_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Source_tagged']])
            target_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Target_Tagged']])
        else:
            source_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Source_Line_Tagged']])
            target_untokenized = ' '.join(
                [pair[0] for pair in wikihow_instance['Target_Line_Tagged']])
        X.append(source_untokenized)
        Y.append(0)
        X.append(target_untokenized)
        Y.append(1)
    return X, Y


def get_XY_from_predefined(path_to_train, path_to_test, path_to_dev):
    Xdev = []
    Ydev = []
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    with open(path_to_train, 'r') as train_json:
        full_train = json.load(train_json)
    with open(path_to_test, 'r') as test_json:
        full_test = json.load(test_json)
    with open(path_to_dev, 'r') as dev_json:
        full_dev = json.load(dev_json)
    Xtrain, Ytrain = get_docs_labels(full_train)
    Xdev, Ydev = get_docs_labels(full_dev)
    Xtest, Ytest = get_docs_labels(full_test)
    return Xtrain, Ytrain, Xdev, Ydev


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, ngram_range_value=(1, 2)):
    """
        Slightly modified from Irshad.
    """
    # vectorize data
    print("Vectorize the data ...")
    count_vec = CountVectorizer(max_features=None, lowercase=False,
                                ngram_range=ngram_range_value, stop_words=None, token_pattern='[^ ]+')
    Xtrain_BOW = count_vec.fit_transform(Xtrain)
    Xdev_BOW = count_vec.transform(Xdev)
    normalize(Xtrain_BOW, copy=False)
    normalize(Xdev_BOW, copy=False)
    print("Train classifier ..")
    classifier = MultinomialNB()
    classifier.fit(Xtrain_BOW, Ytrain)
    print("Finished training ..")
    YpredictDev = classifier.predict_proba(Xdev_BOW)[:, 1]
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


def run_all_noun_types():
    list_of_wikihow_instances = load_json()
    # get everything for different noun modifications
    XtrainDIFF, YtrainDIFF, XdevDIFF, YdevDIFF = get_XY(
        list_of_wikihow_instances)

    # get everything for same noun modifications
    same_list_of_wikihow_instances = load_json(False)
    XtrainSAME, YtrainSAME, XdevSAME, YdevSAME = get_XY(
        same_list_of_wikihow_instances, False, False)

    Xtrain = XtrainDIFF + XtrainSAME
    Ytrain = YtrainDIFF + YtrainSAME
    Xdev = XdevDIFF + XdevSAME
    Ydev = YdevDIFF + YdevSAME

    # split further into X and Y
    train_classifier(Xtrain, Ytrain, Xdev, Ydev)


def get_specific_set_from_data(list_of_wikihow_instances, split='DEV'):
    """
        Parameter of split should be: TRAIN, DEV, TEST 
    """
    requested_set = [
        wikihow_instance for wikihow_instance in list_of_wikihow_instances if wikihow_instance['Loc_in_splits'] == split]
    return requested_set


def get_error_analysis_by_cat(set_to_inspect, list_of_indexes, message):
    """
        set_to_inspect: list of all development/train/test wikihow instances
        list_of_indexes: list with indexes of negative cases or list with indexes of positive cases, as returned by 
        train_classifier()
    """
    freq_dist_entailment = Counter()
    list_of_categories = []
    for index in list_of_indexes:
        correct_instance = set_to_inspect[index]
        # print("{0}\t{1}".format(
        #    correct_instance['Entailment_Rel'], correct_instance['PPDB_Matches']))
        for key, _ in correct_instance['Entailment_Rel'].items():
            list_of_categories.append(correct_instance['Entailment_Rel'][key])
    for rel in list_of_categories:
        freq_dist_entailment[rel] += 1
    res = dict(freq_dist_entailment)
    print(message)
    for key, value in res.items():
        print(key, '\t', value)
    total = sum([value for key, value in res.items()])
    print("TOTAL RELATIONS: ", total)


def main():
    # read json file
    """
    path_to_train = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TRAIN.JSON'
    path_to_dev = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-DEV.JSON'
    path_to_test = './different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TEST.JSON'
    Xtrain, Ytrain, Xdev, Ydev = get_XY_from_predefined(
        path_to_train, path_to_test, path_to_dev)
    positive_cases, negative_cases = train_classifier(Xtrain, Ytrain,
                                                      Xdev, Ydev)

    # remember, predictions are made on the development set.

    with open(path_to_dev, 'r') as json_in:
        development_set = json.load(json_in)
    """

    with open('./same-noun-modifications/same-noun-modifications-keys.json', 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)
    print(list_of_wikihow_instances[0].keys())
    Xtrain, Ytrain, Xdev, Ydev = get_XY(
        list_of_wikihow_instances, use_test_set=False, diff_noun_file=False)

    positive_cases, negative_cases = train_classifier(Xtrain, Ytrain,
                                                      Xdev, Ydev)
    """
    pos_message = "get positive cases"
    get_error_analysis_by_cat(development_set, positive_cases, pos_message)
    print("---------------------------------------------------------")
    neg_message = "get negative cases"
    get_error_analysis_by_cat(development_set, negative_cases, neg_message)
    print(len(Ydev)/2)
    """


if __name__ == '__main__':
    main()
