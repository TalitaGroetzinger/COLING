from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import json
import numpy as np
import gensim
import pickle


def load_json(different_noun_modifications=True):
    if different_noun_modifications:
        path = './classification-data/diff_noun_modifications_PPDB_tagged_with_splits.json'
        print("Using file with DIFF-NOUN-MODIFICATIONS: ", path)
        with open(path, 'r') as json_in:
            list_of_wikihow_instances = json.load(json_in)

    else:
        path = './classification-data/noun_corrections_ppdb_tagged_v3_with_split_info.json'
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
    for source_prediction, target_prediction in zip(YpredictDev[::2], YpredictDev[1::2]):
        if source_prediction < target_prediction:
            positive += 1
        else:
            negative += 1
    accuracy = (positive/(positive+negative))
    print("Accuracy: {0}".format(accuracy))
    get_most_informative_features(classifier, count_vec)


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


def main():
    # read json file
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


if __name__ == '__main__':
    main()
