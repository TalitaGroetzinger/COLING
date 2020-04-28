# This script is used to create dict that also contains the "marked context"

import json
import nltk
from nltk.tokenize import word_tokenize
from progress.bar import Bar
import pickle


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


def process_context(context, line):
    left_context = context['left']
    right_context = context['right']

    if type(context['left']) == str:
        left_context = left_context
    else:
        left_context = ' '.join(left_context)

    if type(context['right']) == str:
        right_context = context['right']
    else:
        right_context = ' '.join(context['right'])

    sentence_in_context = left_context + line + right_context
    return sentence_in_context


def process_context_base_everywhere(source_context, target_line):
    left_context = source_context['left']
    right_context = source_context['right']

    if type(source_context['left']) == str:
        left_context = left_context
    else:
        left_context = ' '.join(left_context)

    if type(source_context['right']) == str:
        right_context = source_context['right']
    else:
        right_context = ' '.join(source_context['right'])

    sentence_in_context = left_context + target_line + right_context
    return sentence_in_context


def regroup_context(context, tokenize=True):
    merged_context = [value if type(value) == str else ' '.join(
        value) for key, value in context.items()]
    if tokenize:
        new_context = ' '.join(merged_context)
        return word_tokenize(new_context)
    else:
        return ' '.join(merged_context)


def format_data(list_of_wikihow_instances):
    # data_dict = {"X_Line": [], "X_Context": [],
    #             "X_Context_Marked": [], "Y": []}
    data_dict = {"X_Line": [], "X_Line_Marked": [], "Y": []}
    bar = Bar("Processing ...", max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()

        matches = wikihow_instance['PPDB_Matches']
        # process everything for source
        source_context = wikihow_instance['Source_Context_5']
        source_line = wikihow_instance['Source_Line']

        source_line_marked = mark_cases(source_line, matches, source=True)

        # prepare to mark context
        # source_context_before_marked = process_context(
        #    source_context, source_line)
        # source_context_marked = mark_cases(
        #    source_context_before_marked, matches, source=True)

        # process everything for target
        target_context = wikihow_instance['Target_Context_5']
        target_line = wikihow_instance['Target_Line']
        target_line_marked = mark_cases(target_line, matches, source=False)

        # prepare to mark context
        # target_context_before_marked = process_context(
        #    target_context, target_line)

        # target_context_before_marked = process_context_base_everywhere(
        #    source_context, target_line)

        # target_context_marked = mark_cases(
        #    target_context_before_marked, matches, source=False)

        # add source components to dict
        data_dict["X_Line"].append(wikihow_instance['Source_Line'])
        data_dict["X_Line_Marked"].append(source_line_marked)

        # data_dict["X_Context"].append(regroup_context(
        #    wikihow_instance["Source_Context_5"], tokenize=False))

        # data_dict["X_Context_Marked"].append(source_context_marked)
        data_dict["Y"].append(0)

        # add target components to dict
        data_dict["X_Line"].append(wikihow_instance['Target_Line'])
        data_dict["X_Line_Marked"].append(target_line_marked)

        # data_dict["X_Context"].append(target_context_before_marked)

        # data_dict["X_Context_Marked"].append(target_context_marked)
        data_dict["Y"].append(1)
    bar.finish()
    return data_dict


def main():
    train, dev, test = read_data()

    print("process train")
    train_dict = format_data(train)

    with open("train_marked_spec_line.pickle", "wb") as pickle_out:
        pickle.dump(train_dict, pickle_out)

    print("process dev")
    dev_dict = format_data(dev)

    with open("dev_marked_spec_line.pickle", "wb") as pickle_out:
        pickle.dump(dev_dict, pickle_out)

    print("process test")
    test_dict = format_data(test)
    with open("test_marked_spec_line.pickle", "wb") as pickle_out:
        pickle.dump(test_dict, pickle_out)


main()
