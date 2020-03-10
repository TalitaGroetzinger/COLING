
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import json


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

    print("Train Samples: ", len(Xtrain))
    print("Dev Samples: ", len(Xdev))
    print("Test Samples: ", len(Xtest))
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def train_classifier(Xtrain, Ytrain, Xdev, Ydev):
    """
        Slightly modified from Irshad.
    """
    # vectorize data
    print("Vectorize the data ...")
    count_vec = CountVectorizer(max_features=None, lowercase=False, ngram_range=(
        1, 2), stop_words=None, token_pattern='[^ ]+')
    Xtrain_BOW = count_vec.fit_transform(Xtrain)
    Xdev_BOW = count_vec.transform(Xdev)
    normalize(Xtrain_BOW, copy=False)
    normalize(Xdev_BOW, copy=False)
    print("Train classifier ..")
    classifier = MultinomialNB()
    classifier.fit(Xtrain_BOW, Ytrain)
    print("Finished training ..")
    YpredictDev = classifier.predict_proba(Xdev_BOW)[:, 1]
    positive = 0.0
    negative = 0.0
    for _, (s, t) in enumerate(zip(YpredictDev[::2], YpredictDev[1::2])):
        if s < t:
            positive += 1.0
        else:
            negative += 1.0
        # print('\t'.join([dev_X[k], dev_X[k+1], str(round(s, 3)), str(round(t, 3)), str(s<t)]))
    accuracy = (positive/(positive+negative))
    print("Accuracy: {0}".format(accuracy))


def main():
    # read data
    with open('./data/noun_corrections_ppdb_tagged_v2.json', 'r') as json_file:
        list_of_wikihow_instances = json.load(json_file)
    X, Y = get_data(list_of_wikihow_instances)
    # split dataset into train, dev, test
    Xtrain, Ytrain, Xdev, Ydev, _, _ = split_data(X, Y)
    train_classifier(Xtrain, Ytrain, Xdev, Ydev)


main()
