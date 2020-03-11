from classifier import get_data
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.pipeline import Pipeline


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
    classifier = LinearSVC()
    classifier.fit(Xtrain_BOW, Ytrain)
    print("Finished training ..")
    YpredictDev = classifier.predict(Xdev)
