import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from features import get_length_features, get_postags, tokenize, pos_tags_and_length, get_length_features_context, coherence_vec, discourse_vec, lrec_vec, lexical_complexity_vec
from sklearn.naive_bayes import MultinomialNB


def get_data(path_to_train, path_to_dev, path_to_test):
    train = pd.read_pickle(path_to_train)
    dev = pd.read_pickle(path_to_dev)
    test = pd.read_pickle(path_to_test)
    return train, dev, test


def train_data(train, dev, test):
    count_vec = TfidfVectorizer(max_features=None, lowercase=False,
                                ngram_range=(1, 2), tokenizer=word_tokenize)
    col_transformer = ColumnTransformer(
        transformers=[
            ('vec1', count_vec, 'X_Line')],  remainder='drop',
        n_jobs=-1,
        sparse_threshold=0)

    print("fit data ")
    Xtrain_fitted = col_transformer.fit_transform(train)
    Xdev_fitted = col_transformer.transform(dev)
    # ------------------------------------------------
    # normalize(Xtrain_fitted)
    # normalize(Xdev_fitted)
    # classification
    print("classify")
    classifier = MultinomialNB()
    classifier.fit(Xtrain_fitted, train["Y"].to_list())

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
    path_to_dev = "noun-modifications/dev-df.pickle"
    path_to_test = "noun-modifications/test-df.pickle"
    path_to_train = "noun-modifications/train-df.pickle"
    train_df, dev_df, test_df = get_data(
        path_to_train, path_to_dev, path_to_test)

    train_data(train_df, dev_df, test_df)


main()
