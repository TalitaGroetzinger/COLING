from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.pipeline import Pipeline, FeatureUnion
from collections import Counter
from progress.bar import Bar
import json
import numpy as np
import gensim
import pickle
import nltk
from features import get_length_features, get_postags, tokenize, pos_tags_and_length, get_length_features_context, coherence_vec, discourse_vec, lrec_vec, lexical_complexity_vec


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


def join_data_match(x):
    x_new = ' '.join(x)
    return x_new.replace('__REV__', '')


def join_data(x):
    return ' '.join(x)


def get_paths():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)
    return path_to_train, path_to_dev, path_to_test


def regroup_context(context):
    merged_context = [value if type(value) == str else ' '.join(value)
                      for key, value in context.items()]
    return ' '.join(merged_context)


def regroup_context_target(target_context, source_context):
    context = []
    left_context = source_context['left']
    current_line = target_context['current']
    right_context = source_context['right']
    context.append(left_context)
    context.append(current_line)
    context.append(right_context)

    new = [elem if type(elem) == str else ' '.join(elem)
           for elem in context]

    return ' '.join(new)


def get_xy(list_of_wikihow_instances, use_context='context'):
    X = []
    Y = []
    for wikihow_instance in list_of_wikihow_instances:
        if use_context == 'context':
            print("use context level")
            # source_context = wikihow_instance['Source_Context_5_Processed']
            # target_context = wikihow_instance['Target_Context_5_Processed']
            source_context = wikihow_instance['Source_Context_5_Processed']
            target_context = wikihow_instance['Target_Context_5_Processed']
            X.append(source_context)
            Y.append(0)
            X.append(target_context)
            Y.append(1)
        elif use_context == 'context-new':
            #print("use new context ... ")
            source_context = regroup_context(
                wikihow_instance['Source_Context_5'])
            # change this line
            # print(source_context)
            target_context = regroup_context_target(
                wikihow_instance['Target_Context_5'], wikihow_instance['Source_Context_5'])
            # print(target_context)
            # print('\n')

            X.append(source_context)
            Y.append(0)
            X.append(target_context)
            Y.append(1)

        else:
            print("use sentence-level")
            source_line = wikihow_instance['Source_Line']
            target_line = wikihow_instance['Target_Line']
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


def regex_tokeniser(x):
    tokenizer = RegexpTokenizer('[^ ]+')
    return tokenizer.tokenize(x)


def train_classifier(Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest):
    # don't forget to remove the __REV__ tags in the from coherence_vec
    # count_vec = CountVectorizer(max_features=None, lowercase=False,
    #                           ngram_range = (1, 2), tokenizer = get_length_features_context, preprocessor = regex_tokeniser)

    count_vec = TfidfVectorizer(max_features=None, lowercase=False,
                                ngram_range=(1, 2), tokenizer=word_tokenize)
    #print("fit data ... ")

    vec = FeatureUnion(
        [
            ('feat', count_vec), ('vec', discourse_vec)
        ]
    )

    Xtrain_fitted = vec.fit_transform(Xtrain)
    Xdev_fitted = vec.transform(Xdev)
    # ------------------------------------------------
    # normalize(Xtrain_fitted)
    # normalize(Xdev_fitted)
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
    print(len(Xdev))
    print(positive+negative)
    print("Accuracy: {0}".format(accuracy))
    get_most_informative_features(classifier, vec)
    return list_of_good_predictions, list_of_bad_predictions


def select_vectorizer(vec='discourse'):
    best_count_vec = TfidfVectorizer(max_features=None, lowercase=False, ngram_range=(1, 2),
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
    train, dev, test = get_paths()

    Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest = get_data(
        train, dev, test, context_value='context-new')

    list_of_good_predictions, list_of_bad_predictions = train_classifier(
        Xtrain, Ytrain, Xdev, Ydev, Xtest, Ytest)


main()
