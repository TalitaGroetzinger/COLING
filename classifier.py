
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import json
import numpy as np


def load_json_file(use_all_versions=True):
    if use_all_versions:
        print("Using all_corrections_wikihow_v6.json")
        with open('./data/all_corrections_wikihow_v6.json', 'r') as json_file:
            list_of_wikihow_instances = json.load(json_file)
    else:
        print("Using noun_corrections_ppdb_tagged_v2.json")
        with open('./data/noun_corrections_ppdb_tagged_v2.json', 'r') as json_file:
            list_of_wikihow_instances = json.load(json_file)
    return list_of_wikihow_instances


def get_data(list_of_wikihow_instances):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        source_untokenized = ' '.join(
            [pair[0] for pair in wikihow_instance['Source_Line_Tagged']])
        target_untokenized = ' '.join(
            [pair[0] for pair in wikihow_instance['Target_Line_Tagged']])
        X.append(source_untokenized)
        Y.append(0)
        X.append(target_untokenized)
        Y.append(1)
    return X, Y


def split_data(X, Y):
    """
        Split data into a train, dev and test set.
    """
    # get a train and test set (keep random state same for reproducability)
    Xtrain, Xtest_first, Ytrain, Ytest_first = train_test_split(
        X, Y, test_size=0.4, random_state=1)

    # split test set further into test and development (keep random state same for reproducability)
    Xtest, Xdev, Ytest, Ydev = train_test_split(
        Xtest_first, Ytest_first, test_size=0.2, random_state=1)

    assert len(Xtrain) == len(Ytrain)
    assert len(Xdev) == len(Ydev)
    assert len(Xtest) == len(Ytest)
    print("------------------------------------")
    print("Train Samples: ", len(Xtrain))
    print("Dev Samples: ", len(Xdev))
    print("Test Samples: ", len(Xtest))
    print("------------------------------------")
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, ngram_range_value=(1, 1)):
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
    # read data
    list_of_wikihow_instances = load_json_file(True)
    X, Y = get_data(list_of_wikihow_instances)
    # split dataset into train, dev, test
    Xtrain, Ytrain, Xdev, Ydev, _, _ = split_data(X, Y)
    train_classifier(Xtrain, Ytrain, Xdev, Ydev)


main()
