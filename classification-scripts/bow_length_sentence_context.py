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
    if length_context > 174:
        length_type = "long"
    else:
        length_type = "short"
    doc = [token+'_' +
           length_type for token in word_tokenize(line)]
    return doc


def get_length(x):
    x = word_tokenize(x)
    return len(x)


def add_to_dict(wikihow_dict):
    length_features = []
    all_lens = []
    bar = Bar("Processing ...", max=len(wikihow_dict['X_Line']))
    for i in range(len(wikihow_dict['X_Line'])):
        bar.next()
        context_length = get_length(wikihow_dict['X_Context_base'][i])
        all_lens.append(context_length)
        doc = special_tokenizer(wikihow_dict['X_Line'][i], context_length)
        length_features.append(doc)
    bar.finish()

    wikihow_dict["X_Context_len"] = length_features
    assert len(wikihow_dict["X_Context_len"]
               ) == len(wikihow_dict['X_Line'])
    print(np.mean(all_lens))
    return wikihow_dict


def main():
    # open the files

    with open("./train_dict.pickle", "rb") as pickle_in:
        train = pickle.load(pickle_in)
    with open("./dev_dict.pickle", "rb") as pickle_in:
        dev = pickle.load(pickle_in)
    with open("./test_dict.pickle", "rb") as pickle_in:
        test = pickle.load(pickle_in)

    # do same for train

    dev_df = add_to_dict(dev)

    test_df = add_to_dict(test)

    train_df = add_to_dict(train)

    with open("./noun-modifications/train_tok_alength.pickle", "wb") as pickle_in:
        pickle.dump(train_df, pickle_in)
    with open("./noun-modifications/dev_tok_alength.pickle", "wb") as pickle_in:
        pickle.dump(dev_df, pickle_in)
    with open("./noun-modifications/test_tok_alength", "wb") as pickle_in:
        pickle.dump(test_df, pickle_in)

    print(test_df["X_Context_len"][0:100])


main()
