# This script was used to make a dev, test and train set with a special column
# called X_Context_Length. This contains all the tokenized source or sentence lines and
# the length of the context. The instances can then directly be feed into the Tfidftokenizer.


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
from features import MultipleItemSelector


def special_tokenizer(line, article_length_value):
    """
      Parameters
      ----------
      line = X_line, untokenized
      X_Context = X_Context, tokenized using word_tokenized
    """
    # compute the length of context
    length_context = article_length_value
    if length_context > 1309.4241171540534:
        length_type = "long"
    else:
        length_type = "short"
    doc = [token+'_' +
           length_type for token in word_tokenize(line)]
    return doc


def add_to_dict(wikihow_dict):
    length_features = []
    bar = Bar("Processing ...", max=len(wikihow_dict['X_Line']))
    for i in range(len(wikihow_dict['X_Line'])):
        bar.next()
        doc = special_tokenizer(
            wikihow_dict['X_Line'][i], wikihow_dict['X_article_info'][i]['type_token_ratio'])
        length_features.append(doc)
    bar.finish()

    wikihow_dict["X_in_context_length"] = length_features
    assert len(wikihow_dict["X_in_context_length"]
               ) == len(wikihow_dict['X_Line'])
    return wikihow_dict


def main():

    # open the files

    with open("./noun-modifications/train_article.pickle", "rb") as pickle_in:
        train = pickle.load(pickle_in)
    with open("./noun-modifications/dev_article.pickle", "rb") as pickle_in:
        dev = pickle.load(pickle_in)
    with open("./noun-modifications/test_article.pickle", "rb") as pickle_in:
        test = pickle.load(pickle_in)

    # do same for train

    dev_df = add_to_dict(dev)

    test_df = add_to_dict(test)
    train_df = add_to_dict(train)

    print(dev_df.keys())
    for i in range(1):
        print(dev_df['X_in_context_length'][i])
        print(dev_df['X_Line'][i])

    with open("./noun-modifications/train_tok_alength.pickle", "wb") as pickle_in:
        pickle.dump(train_df, pickle_in)
    with open("./noun-modifications/dev_tok_alength.pickle", "wb") as pickle_in:
        pickle.dump(dev_df, pickle_in)
    with open("./noun-modifications/test_tok_alength", "wb") as pickle_in:
        pickle.dump(test_df, pickle_in)


main()