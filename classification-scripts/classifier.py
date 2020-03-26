from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


def get_docs_labels_context(list_of_wikihow_instances):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        source_context = wikihow_instance['Source_Context_5_Processed']
        target_context = wikihow_instance['Target_Context_5_Processed']
        X.append(source_context)
        Y.append(0)
        X.append(target_context)
        Y.append(1)
    return X, Y


def get_xy(path_to_train, path_to_test, path_to_dev, different_nouns=True, context=True):
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

    if context:
        print("Run classifier on context ... ")
        Xtrain, Ytrain = get_docs_labels_context(full_train)
        Xtest, Ytest = get_docs_labels_context(full_test)
        Xdev, Ydev = get_docs_labels_context(full_dev)
    else:
        print("Do not use context ...")
        if different_nouns:
            Xtrain, Ytrain = get_docs_labels(full_train)
            Xdev, Ydev = get_docs_labels(full_dev)
            Xtest, Ytest = get_docs_labels(full_test)
        else:
            Xtrain, Ytrain = get_docs_labels(full_train, False)
            Xdev, Ydev = get_docs_labels(full_dev, False)
            Xtest, Ytest = get_docs_labels(full_test, False)
    return Xtrain, Ytrain, Xdev, Ydev


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, ngram_range_value=(1, 2)):
    """
        Slightly modified from Irshad.
    """
    # vectorize data
    print("Vectorize the data ...")
    # count_vec = CountVectorizer(max_features=None, lowercase=False,
    #                            ngram_range=ngram_range_value, stop_words=None, token_pattern='[^ ]+')

    count_vec_1 = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=ngram_range_value,
                                  tokenizer=get_length_features, preprocessor=word_tokenize)
    count_vec_2 = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=ngram_range_value,
                                  tokenizer=get_postags, preprocessor=word_tokenize)

    count_vec = FeatureUnion([('count1', count_vec_1), ('count2', count_vec_2)


                              ])

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


def run_all_noun_types(path_to_train_same, path_to_dev_same, path_to_test_same, path_to_train_diff, path_to_dev_diff, path_to_test_diff):

    # get everything for different noun modifications
    XtrainDIFF, YtrainDIFF, XdevDIFF, YdevDIFF = get_xy(
        path_to_train_diff, path_to_test_diff, path_to_dev_diff, different_nouns=True, context=False)

    # get everything for same noun modifications
    XtrainSAME, YtrainSAME, XdevSAME, YdevSAME = get_xy(
        path_to_train_same, path_to_test_same, path_to_dev_same, different_nouns=False, context=False)

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
    # read json file

    path_to_train_same, path_to_dev_same, path_to_test_same = get_paths(False)
    path_to_train_diff, path_to_dev_diff, path_to_test_diff = get_paths(True)

    run_all_noun_types(path_to_train_same, path_to_dev_same, path_to_test_same,
                       path_to_train_diff, path_to_dev_diff, path_to_test_diff)

    # def get_xy(path_to_train, path_to_test, path_to_dev, different_nouns=True, context=True):
    # Xtrain, Ytrain, Xdev, Ydev = get_xy(
    #    path_to_train_same, path_to_test_same, path_to_dev_same, different_nouns=False, context=False)

    # positive_cases, negative_cases = train_classifier(Xtrain, Ytrain,
    #                                                  Xdev, Ydev)

    # remember, predictions are made on the development set.

    with open(path_to_dev_same, 'r') as json_in:
        development_set = json.load(json_in)

    pos_message = "get positive cases"
    get_error_analysis_by_cat(development_set, positive_cases, pos_message)
    print("---------------------------------------------------------")
    neg_message = "get negative cases"
    get_error_analysis_by_cat(development_set, negative_cases, neg_message)


if __name__ == '__main__':
    main()
