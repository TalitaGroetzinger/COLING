import json
import pandas as pd
from progress.bar import Bar
from nltk.tokenize import word_tokenize, sent_tokenize
from features_for_pytorch import type_token_ratio, check_discourse_matches
from similarity import compute_sentence_similarity
import nltk
import numpy as np
from features_for_pytorch import mark_cases


def add_length(context, average=False):
    if average:
        sentences = sent_tokenize(context)
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        return avg_sentence_length

    else:
        context_length = word_tokenize(context)
        if len(context_length) >= 150:
            return 1
        else:
            return 0


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


def process_context_sim(context, left_side=False):

    if left_side:
        left_context = context['left']
        left_context = context['left']
        if type(context['left']) == str:
            try:
                return nltk.sent_tokenize(context['left'])[-1]
            except IndexError:
                return "UNK"

        else:
            try:
                return left_context[-1]  # return the last sentence
            except IndexError:
                return "UNK"
    else:
        right_context = context['right']
        right_context = context['right']
        if type(context['right']) == str:
            try:
                return nltk.sent_tokenize(context['right'])[0]
            except IndexError:
                return "UNK"

        else:
            try:
                return right_context[0]  # return the last sentence
            except IndexError:
                return "UNK"


def read_data():
    path_to_dir = '../classification-scripts/noun-modifications'
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


def process_dict(list_of_wikihow_instances, json_to_write_filename):
    collection = []
    bar = Bar("Processing", max=len(list_of_wikihow_instances))
    index_for_source = 0
    index_for_target = 1
    for c, wikihow_instance in enumerate(list_of_wikihow_instances, 1):
        bar.next()
        source_row = {}
        target_row = {}

        # -----------------------------
        # process everything for source
        # -----------------------------
        source_row["Filename"] = wikihow_instance["Filename"]
        source_row["Line"] = wikihow_instance["Source_Line"]
        source_row["Label"] = "0"
        source_row["Context"] = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])

        # add type token ratio of the source_context
        source_context = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])

        # add ttr, discourse, sim and length
        source_row["TTR"] = type_token_ratio(source_context)
        source_row["Discourse_count"] = check_discourse_matches(source_context)[
            'score']
        last_sent_left = process_context_sim(
            wikihow_instance["Source_Context_5"], left_side=True)
        last_sent_right = process_context_sim(
            wikihow_instance["Source_Context_5"], left_side=False)

        source_row["Cos_sim_left"] = compute_sentence_similarity(
            wikihow_instance["Source_Line"], last_sent_left)
        source_row["Cos_sim_right"] = compute_sentence_similarity(
            wikihow_instance["Source_Line"], last_sent_right)

        source_row["Length"] = add_length(source_context)
        source_row["Length_avg"] = add_length(source_context, average=True)

        source_row["Noun_overlap"] = mark_cases(
            source_context, wikihow_instance["PPDB_Matches"], source=True)

        # repeat for base context everywhere

        source_row["Length_base_exp"] = add_length(source_context)
        source_row["Length_avg_base_exp"] = add_length(
            source_context, average=True)
        source_row["Cos_sim_left_base_exp"] = compute_sentence_similarity(
            wikihow_instance["Source_Line"], last_sent_left)
        source_row["Cos_sim_right_base_exp"] = compute_sentence_similarity(
            wikihow_instance["Source_Line"], last_sent_right)

        source_row["TTR_base_exp"] = type_token_ratio(source_context)
        source_row["Discourse_count_base_exp"] = check_discourse_matches(
            source_context)['score']
        source_row["Noun_overlap_base_exp"] = source_row["Noun_overlap"]
        source_row["ID"] = index_for_source

        # ------------------------------------------------------
        # process everything for target
        # ------------------------------------------------------

        # get everything for normal set-up
        target_context = process_context(
            wikihow_instance["Target_Context_5"], wikihow_instance["Target_Line"])
        target_row["Filename"] = wikihow_instance["Filename"]
        target_row["Line"] = wikihow_instance["Target_Line"]
        target_row["Label"] = "1"
        target_row["Context"] = process_context(
            wikihow_instance["Target_Context_5"], wikihow_instance["Target_Line"])

        # get length, discourse, trr,

        target_row["Length"] = add_length(target_context)
        target_row["Length_avg"] = add_length(target_context, average=True)

        target_row["Discourse_count"] = check_discourse_matches(target_context)[
            'score']
        target_row["Cos_sim_left"] = compute_sentence_similarity(
            wikihow_instance["Target_Line"], process_context_sim(wikihow_instance["Target_Context_5"], left_side=True))
        target_row["Cos_sim_right"] = compute_sentence_similarity(
            wikihow_instance["Target_Line"], process_context_sim(wikihow_instance["Target_Context_5"], left_side=False))

        target_context = process_context(
            wikihow_instance["Target_Context_5"], wikihow_instance["Target_Line"])
        target_row["TTR"] = type_token_ratio(target_context)
        target_row["Length"] = add_length(target_context)
        target_row["Noun_overlap"] = mark_cases(
            target_context, wikihow_instance["PPDB_Matches"], source=False)

        # add type token ratio, length, discourse for base context everywhere set-up
        target_context_base = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Target_Line"])
        target_row["TTR_base_exp"] = type_token_ratio(target_context_base)
        target_row["Length_base_exp"] = add_length(target_context_base)
        target_row["Length_avg_base_exp"] = add_length(
            target_context_base, average=True)
        target_row["Discourse_count_base_exp"] = check_discourse_matches(process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Target_Line"]))['score']
        target_row["Cos_sim_left_base_exp"] = compute_sentence_similarity(
            wikihow_instance["Target_Line"], process_context_sim(wikihow_instance["Source_Context_5"], left_side=True))
        target_row["Cos_sim_right_base_exp"] = compute_sentence_similarity(
            wikihow_instance["Target_Line"], process_context_sim(wikihow_instance["Source_Context_5"], left_side=False))

        # source = false want je gebruikt de target_line
        target_row["Noun_overlap_base_exp"] = mark_cases(
            target_context_base, wikihow_instance["PPDB_Matches"], source=False)

        target_row["ID"] = index_for_target
        collection.append(source_row)
        collection.append(target_row)
        index_for_target = index_for_target + 2
        index_for_source = index_for_source + 2

    bar.finish()
    # convert to dataframe
    df = pd.DataFrame(collection)

    df.to_json(json_to_write_filename, orient='records', lines=True)


def main():
    train_set, dev_set, test_set = read_data()

    # process test set
    process_dict(test_set,
                 "test_set_pytorch_discourse_sim_ttr_length_right_sim_noun_overlap_new.json")

    process_dict(
        dev_set, "dev_set_pytorch_discourse_sim_ttr_length_right_sim_new.json")
    process_dict(
        train_set, "train_set_pytorch_discourse_sim_ttr_length_right_sim_new.json")


if __name__ == "__main__":
    main()
