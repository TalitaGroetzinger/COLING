import json
from progress.bar import Bar
import pickle
import nltk
from nltk.tokenize import word_tokenize


def mark_cases(context, matches, source=True):
    if source:
        match = [elem[0][0].lower() for elem in matches]
    else:
        match = [elem[1][0].lower() for elem in matches]

    tokenized_doc = word_tokenize(context)
    document = [word[0] for word in nltk.pos_tag(tokenized_doc)]

    final_rep = []
    for word in document:
        if word.lower() in match:
            word = word + "__REV__"
            final_rep.append(word)
        else:
            final_rep.append(word)
    return final_rep


def read_data():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)

    with open(path_to_test, 'r') as json_in_test:
        test = json.load(json_in_test)

    with open(path_to_train, 'r') as json_in_train:
        train = json.load(json_in_train)

    with open(path_to_dev, 'r') as json_in_dev:
        dev = json.load(json_in_dev)
    return train, dev, test


def process_context(context, current_line):
    left_context = context['left']
    right_context = context['right']

    if type(context['left']) == str:
        left_context_plus_current = context['left'] + ' ' + current_line
    else:
        left_context_plus_current = ' '.join(
            context['left']) + ' ' + current_line

    if type(context['right']) == str:
        context = left_context_plus_current + ' ' + context['right']
    else:
        context = left_context_plus_current + ' ' + ' '.join(context['right'])

    return context


def format_data(list_of_wikihow_instances):
    data_dict = {"X_Line": [], "X_Context": [],
                 "X_Context_base": [], "X_Context_marked": [], "X_Context_marked_base": [],  "Y": []}
    bar = Bar("Processing ...", max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()

        # process everything for source
        source_context = wikihow_instance['Source_Context_5']
        source_line = wikihow_instance['Source_Line']
        source_context_processed = process_context(source_context, source_line)

        matches = wikihow_instance['PPDB_Matches']
        source_context_marked = mark_cases(
            source_context_processed, matches, source=True)

        data_dict["X_Line"].append(source_line)
        data_dict["X_Context"].append(source_context_processed)
        data_dict["X_Context_base"].append(source_context_processed)
        data_dict["X_Context_marked"].append(source_context_marked)
        data_dict["X_Context_marked_base"].append(source_context_marked)

        data_dict["Y"].append(0)

        # process everything for target
        target_line = wikihow_instance['Target_Line']
        target_context = wikihow_instance['Target_Context_5']
        target_context_processed = process_context(target_context, target_line)

        # use the source context
        target_context_base = process_context(source_context, target_line)

        target_context_marked = mark_cases(
            target_context_processed, matches, source=False)

        target_context_marked_base = mark_cases(
            target_context_base, matches, source=False)

        data_dict["X_Line"].append(target_line)
        data_dict["X_Context"].append(target_context_processed)

        # append special context for target
        data_dict["X_Context_base"].append(target_context_base)
        data_dict["X_Context_marked"].append(target_context_marked)
        data_dict["X_Context_marked_base"].append(target_context_marked_base)

        data_dict["Y"].append(1)
    bar.finish()
    return data_dict


def main():
    train, dev, test = read_data()

    train_dict = format_data(train)
    dev_dict = format_data(dev)
    test_dict = format_data(test)

    with open("dev_dict.pickle", "wb") as pickle_out:
        pickle.dump(dev_dict, pickle_out)
    with open("test_dict.pickle", "wb") as pickle_out:
        pickle.dump(test_dict, pickle_out)
    with open("train_dict.pickle", "wb") as pickle_out:
        pickle.dump(train_dict, pickle_out)


main()
