import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
import pandas as pd
from nltk.tokenize import word_tokenize
from features import *
from sklearn.naive_bayes import MultinomialNB
import pickle
from progress.bar import Bar


def regroup_context(context, tokenize=True):
    merged_context = [value if type(value) == str else ' '.join(
        value) for key, value in context.items()]
    if tokenize:
        new_context = ' '.join(merged_context)
        return word_tokenize(new_context)
    else:
        return new_context


def get_paths():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-length.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-length.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-length.json'.format(
        path_to_dir)
    return path_to_train, path_to_dev, path_to_test


def make_df(json_file):
    df_dict = {"X_Line": [], "X_article_info": [], "Y": []}
    bar = Bar("Processing ...", max=len(json_file))
    for wikihow_instance in json_file:
        bar.next()
        # add the source components
        df_dict["X_Line"].append(wikihow_instance['Source_Line'])

        type_token_ratio_source = wikihow_instance['Source_Article_info']['Length']

        df_dict["X_article_info"].append(
            {"type_token_ratio": type_token_ratio_source})

        df_dict["Y"].append(0)
        # add the target components
        df_dict["X_Line"].append(wikihow_instance['Target_Line'])

        type_token_ratio_target = wikihow_instance['Target_Article_info']['Length']

        df_dict["X_article_info"].append(
            {"type_token_ratio": type_token_ratio_target})

        df_dict["Y"].append(1)
    bar.finish()
    return df_dict


def get_average(dict_file):
    lens = []

    for elem in dict_file['X_article_info']:
        lens.append(elem['type_token_ratio'])

    mean = np.mean(lens)
    print(mean)

    len_info = []
    for elem in dict_file['X_article_info']:
        if elem['type_token_ratio'] > mean:
            value = 'long'
        else:
            value = 'short'
        len_info.append({"article_length": value})

    dict_file['X_article_info_new'] = len_info
    return dict_file


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


def dummy(x):
    return x


def train_data(train, dev, test):
    """
    count_vec = TfidfVectorizer(max_features=None, lowercase=False,
                                ngram_range=(1, 2), tokenizer=word_tokenize)

    count_vec_pipe = Pipeline([('selector', ItemSelector(key='X_Line')
                                ), ('tfidf', count_vec)])

    article_length_vec = Pipeline([('selector', ItemSelector(key='X_article_info_new')
                                    ), ('length', DictVectorizer())])

    vec = FeatureUnion(
        [('vec1', count_vec_pipe), ('vec2', article_length_vec)])
    """

    vec = Pipeline([('selector', ItemSelector(key='X_in_context_length')
                     ), ('tfidf', TfidfVectorizer(preprocessor=dummy, tokenizer=dummy, ngram_range=(1, 2)))])

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
    """
    with open("noun-modifications/train_tok_new.pickle", "rb") as train_in:
        train = pickle.load(train_in)

    with open("noun-modifications/dev_tok_new.pickle", "rb") as dev_in:
        dev = pickle.load(dev_in)

    with open("./noun-modifications/test_tok_new.pickle", "rb") as test_in:
        test = pickle.load(test_in)


    path_to_train, path_to_dev, path_to_test = get_paths()

    with open(path_to_train, 'r') as json_in:
        train_open = json.load(json_in)

    with open(path_to_dev, 'r') as json_in:
        dev_open = json.load(json_in)

    with open(path_to_test, 'r') as json_in:
        test_open = json.load(json_in)

    train = make_df(train_open)

    train = get_average(train)

    dev = make_df(dev_open)
    dev = get_average(dev)

    test = make_df(test_open)
    test = get_average(test)


    with open("./noun-modifications/train_article.pickle", "wb") as pickle_in:
        pickle.dump(train, pickle_in)
    with open("./noun-modifications/dev_article.pickle", "wb") as pickle_in:
        pickle.dump(dev, pickle_in)
    with open("./noun-modifications/test_article.pickle", "wb") as pickle_in:
        pickle.dump(test, pickle_in)

    #train_data(train, dev, test)
    """

    with open("./noun-modifications/train_tok_alength.pickle", "rb") as pickle_in:
        train = pickle.load(pickle_in)
    with open("./noun-modifications/dev_tok_alength.pickle", "rb") as pickle_in:
        dev = pickle.load(pickle_in)
    with open("./noun-modifications/test_tok_alength", "rb") as pickle_in:
        test = pickle.load(pickle_in)

    train_data(train, dev, test)


main()
