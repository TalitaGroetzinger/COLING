from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from progress.bar import Bar
import json
import numpy as np
import gensim
import pickle
import nltk
from features import get_length_features, get_postags, tokenize, pos_tags_and_length, get_length_features_context, coherence_vec, discourse_vec, lrec_vec


def mark_cases(context, matches, source=True):
    if source:
        match = [elem[0][0].lower() for elem in matches]
    else:
        match = [elem[1][0].lower() for elem in matches]

    document = []
    for sent in context:
        tokenized = word_tokenize(sent)
        tokenized = [word[0] for word in nltk.pos_tag(tokenized)]
        for word in tokenized:
            document.append(word)

    final_rep = []
    for word in document:
        if word.lower() in match:
            word = word + "__REV__"
            final_rep.append(word)
        else:
            final_rep.append(word)
    return final_rep


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


def join_data(x):
    return ' '.join(x)


def get_paths(use_all=True):
    if use_all:
        path_to_dir = '../classification-scripts/noun-modifications'
        path_to_train = "{0}/noun-modifications-train-5-new.json".format(
            path_to_dir)
        path_to_dev = "{0}/noun-modifications-dev-5-new.json".format(
            path_to_dir)
        path_to_test = "{0}/noun-modifications-test-5-new.json".format(
            path_to_dir)

        return path_to_train, path_to_dev, path_to_test
    else:
        path_to_dir_diff = '../classification-scripts/different-noun-modifications'
        path_to_train_diff = '{0}/DIFF-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'.format(
            path_to_dir_diff)
        path_to_dev_diff = '{0}/different-noun-modifications/DIFF-NOUN-MODIFICATIONS-DEV-5-new.JSON'.format(
            path_to_dir_diff)
        path_to_test_diff = '{0}/DIFF-NOUN-MODIFICATIONS-TEST-5-new.JSON'.format(
            path_to_dir_diff)

        # get same-noun modifications
        path_to_dir_same = '../classification-scripts/same-noun-modifications'
        path_to_train_same = '{0}/SAME-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'.format(
            path_to_dir_same)
        path_to_dev_same = '{0}/SAME-NOUN-MODIFICATIONS-DEV-5-new.JSON'.format(
            path_to_dir_same)
        path_to_test_same = '{0}/SAME-NOUN-MODIFICATIONS-TEST-5-new.JSON'.format(
            path_to_dir_same)

        return path_to_train_diff, path_to_dev_diff, path_to_test_diff, path_to_train_same, path_to_dev_same, path_to_test_same


def get_xy(list_of_wikihow_instances, use_context='context'):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        if use_context == 'context':
            print("use context level")
            source_context = wikihow_instance['Source_Context_5_Processed']
            target_context = wikihow_instance['Target_Context_5_Processed']
            X.append(source_context)
            Y.append(0)
            X.append(target_context)
            Y.append(1)
        elif use_context == 'context-matches':
            print("use context plus matches")
            source_context = wikihow_instance['Source_Context_5_Processed']
            target_context = wikihow_instance['Target_Context_5_Processed']
            matches = wikihow_instance['PPDB_Matches']

            new_source = mark_cases(source_context, matches, source=True)
            new_target = mark_cases(target_context, matches, source=False)

            X.append(new_source)
            Y.append(0)
            X.append(new_target)
            Y.append(1)

        else:
            print("use sentence-level")
            source_line = ' '.join(
                [pair[0] for pair in wikihow_instance['Source_Line_Tagged']])
            target_line = ' '.join(
                [pair[0] for pair in wikihow_instance['Target_Line_Tagged']])
            X.append(source_line)
            Y.append(0)
            X.append(target_line)
            Y.append(1)
    assert len(X) == len(Y)
    return X, Y


def get_data(path_to_train, path_to_dev, path_to_test, context_value='context'):
    """
        use_context_level: 'context', 'context-matches', 'sentence'
    """
    # load train and test set
    with open(path_to_train, 'r') as json_in:
        train = json.load(json_in)
    with open(path_to_dev, 'r') as json_in:
        dev = json.load(json_in)
    with open(path_to_test, 'r') as json_in:
        test = json.load(json_in)

    Xtrain, Ytrain = get_xy(train, use_context=context_value)
    Xdev, Ydev = get_xy(dev, use_context=context_value)
    Xtest, Ytest = get_xy(test, use_context=context_value)
    return Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest):
    # hier moet de dict van Xtrain dus in
    # fit to countvec
    count_vec = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 2),
                                token_pattern='[^ ]+', preprocessor=join_data)
    """
    Xtrain_fitted = count_vec.fit_transform(Xtrain)
    Xdev_fitted = count_vec.transform(Xdev)
    """
    print("fit data ... ")
    vec = FeatureUnion(
        [
            ('feat', discourse_vec), ('vec', count_vec)
        ]
    )

    Xtrain_fitted = vec.fit_transform(Xtrain)
    Xdev_fitted = vec.transform(Xdev)
    # ------------------------------------------------

    # classification
    classifier = MultinomialNB()
    classifier.fit(Xtrain_fitted, Ytrain)

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
    print("Accuracy: {0}".format(accuracy))
    get_most_informative_features(classifier, vec)
    return list_of_good_predictions, list_of_bad_predictions


def select_vectorizer(vec='discourse'):
    best_count_vec = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 2),
                                     token_pattern='[^ ]+', preprocessor=join_data)
    length_features = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 2),
                                      token_pattern='[^ ]+', preprocessor=join_data)
    if vec == 'discourse':
        vec = FeatureUnion(
            [
                ('feat', discourse_vec), ('vec', best_count_vec)
            ]
        )
    elif vec == 'word-overlap':
        vec = FeatureUnion(
            [
                ('feat', coherence_vec), ('vec', best_count_vec)
            ]
        )
    else:
        vec = lrec_vec
    return vec


def main():
    # get different nouns
    # use_all=false to get path_to_train_diff, path_to_dev_diff, path_to_test_diff, path_to_train_same, path_to_dev_same, path_to_test_same
    train, dev, test = get_paths(use_all=True)

    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = get_data(train, dev, test)

    # list_of_good_predictions, list_of_bad_predictions = train_classifier(
    #    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest)


main()
