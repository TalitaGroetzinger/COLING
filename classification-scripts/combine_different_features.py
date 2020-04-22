import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.tokenize import word_tokenize
from features import get_length_features, get_postags, tokenize, pos_tags_and_length, get_length_features_context, coherence_vec, discourse_vec, lrec_vec, lexical_complexity_vec
from sklearn.naive_bayes import MultinomialNB
import pickle
from progress.bar import Bar


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def regroup_context(context, tokenize=True):
    merged_context = [value if type(value) == str else ' '.join(
        value) for key, value in context.items()]
    if tokenize:
        new_context = ' '.join(merged_context)
        return [word_tokenize(sent) for sent in new_context]
    else:
        return new_context


def get_paths():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)
    return path_to_train, path_to_dev, path_to_test


def make_df(json_file):
    df_dict = {"X_Line": [], "X_Context": [], "Y": []}
    bar = Bar("Processing ...", max=len(json_file))
    for wikihow_instance in json_file:
        bar.next()
        # add the source components
        df_dict["X_Line"].append(wikihow_instance['Source_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Source_Context_5'], tokenize=False))
        df_dict["Y"].append(0)
        # add the target components
        df_dict["X_Line"].append(wikihow_instance['Target_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Target_Context_5'], tokenize=False))
        df_dict["Y"].append(1)
    bar.finish()
    return df_dict


def make_df_save(json_file, name_to_write):
    df_dict = {"X_Line": [], "X_Context": [], "Y": []}
    bar = Bar("Processing ...", max=len(json_file))
    for wikihow_instance in json_file:
        bar.next()
        # add the source components
        df_dict["X_Line"].append(wikihow_instance['Source_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Source_Context_5'], tokenize=True))
        df_dict["Y"].append(0)
        # add the target components
        df_dict["X_Line"].append(wikihow_instance['Target_Line'])
        df_dict["X_Context"].append(regroup_context(
            wikihow_instance['Target_Context_5'], tokenize=True))
        df_dict["Y"].append(1)
    bar.finish()
    with open(name_to_write, 'wb') as pickle_out:
        pickle.dump(df_dict, pickle_out)


def train_data(train, dev, test):
    count_vec = TfidfVectorizer(max_features=None, lowercase=False,
                                ngram_range=(1, 2), tokenizer=word_tokenize)

    vec1 = Pipeline([
        ('selector', ItemSelector(key='X_Line')
         ), ('count_vec', count_vec),
    ])

    vec2 = Pipeline([
        ('selector', ItemSelector(key='X_Context')
         ), ('count_vec', lexical_complexity_vec),
    ])
    vec = FeatureUnion([('vec1', vec1), ('vec2', vec2)])
    print("fit data ")
    Xtrain_fitted = vec.fit_transform(train)
    Xdev_fitted = vec.transform(dev)
    # ------------------------------------------------
    # normalize(Xtrain_fitted)
    # normalize(Xdev_fitted)
    # classification
    print("classify")
    classifier = MultinomialNB()
    classifier.fit(Xtrain_fitted, train["Y"])

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
    print(len(dev))
    print(positive+negative)
    print("Accuracy: {0}".format(accuracy))
    return list_of_good_predictions, list_of_bad_predictions


def main():
    path_to_train, path_to_dev, path_to_test = get_paths()
    with open(path_to_dev, 'r') as json_in:
        dev = json.load(json_in)
    with open(path_to_train, 'r') as json_in:
        train = json.load(json_in)
    with open(path_to_test, 'r') as json_in:
        test = json.load(json_in)

    print("make one for dev")
    make_df_save(dev, "dev_tok.pickle")
    print("make one for train")
    make_df_save(train, "train_tok.pickle")
    print("make one for test")

    make_df_save(test, "test_tok.pickle")

#train_data(train, dev, test)


main()
