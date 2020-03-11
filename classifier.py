from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import json
import numpy as np
import gensim
import pickle


def load_json_file(use_all_versions=False):
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


def split_data_by_article(list_of_wikihow_instances):
    """
        Split data into a train, dev and test set based on filename. 
    """
    list_of_filenames = list(set([wikihow_instance['Filename']
                                  for wikihow_instance in list_of_wikihow_instances]))
    train_set_files, test_set_files = train_test_split(
        list_of_filenames, shuffle=False, train_size=0.8, random_state=1)
    test_set_files, dev_set_files = train_test_split(
        test_set_files, train_size=0.5, random_state=1, shuffle=False)

    print("Articles in train set: ", len(train_set_files))
    print("Articles in dev set: ", len(dev_set_files))
    print("Articles in test set: ", len(test_set_files))
    return train_set_files, dev_set_files, test_set_files


def get_XY(list_of_wikihow_instances, train_set_files, dev_set_files, test_set_files, use_test_set=False):
    Xdev = []
    Ydev = []
    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []
    for wikihow_instance in list_of_wikihow_instances:
        source_untokenized = ' '.join(
            [pair[0] for pair in wikihow_instance['Source_Line_Tagged']])
        target_untokenized = ' '.join(
            [pair[0] for pair in wikihow_instance['Target_Line_Tagged']])
        if wikihow_instance['Filename'] in train_set_files:
            Xtrain.append(source_untokenized)
            Ytrain.append(0)
            Xtrain.append(target_untokenized)
            Ytrain.append(1)
        elif wikihow_instance['Filename'] in dev_set_files:
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
    count_vec = TfidfVectorizer(max_features=None, lowercase=False,
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
    list_of_wikihow_instances = load_json_file(False)
    """
    # get splits by article
    train_set_files, dev_set_files, test_set_files = split_data_by_article(
        list_of_wikihow_instances)

    with open("test_set_files.pickle", "wb") as pickle_in:
        pickle.dump(test_set_files, pickle_in)
    with open("train_set_files.pickle", "wb") as pickle_in:
        pickle.dump(train_set_files, pickle_in)
    with open("dev_set_files.pickle", "wb") as pickle_in:
        pickle.dump(dev_set_files, pickle_in)

    # pickle.dump(test_set_files, ))
    #pickle.dump(train_set_files, open("train_set_files.pickle", "wb"))
    #pickle.dump(dev_set_files, open("dev_set_files.pickle", "wb"))
    """
    train_set_files = pickle.load(open("train_set_files.pickle", "rb"))
    dev_set_files = pickle.load(open("dev_set_files.pickle", "rb"))
    test_set_files = pickle.load(open("test_set_files.pickle", "rb"))
    # split further into X and Y
    Xtrain, Ytrain, Xdev, Ydev = get_XY(
        list_of_wikihow_instances, train_set_files, dev_set_files, test_set_files, use_test_set=False)
    train_classifier(Xtrain, Ytrain, Xdev, Ydev)


if __name__ == '__main__':
    main()
