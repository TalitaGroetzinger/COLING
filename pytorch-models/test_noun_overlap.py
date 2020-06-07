import json
import pandas as pd
from progress.bar import Bar
from nltk.tokenize import word_tokenize, sent_tokenize
from features_for_pytorch import type_token_ratio, check_discourse_matches
from similarity import compute_sentence_similarity
import nltk
import numpy as np
from process_data_for_pytorch import read_data, process_context
from nltk.tokenize import word_tokenize
from collections import Counter


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
    coherence_score = compute_coherence(final_rep)
    return final_rep, coherence_score


def compute_coherence(doc):
    """
        Input: X formatted with __REV__
    """

    freq = Counter()
    d = {}
    for word in doc:
        if '__REV__' in word:
            freq[word.lower()] += 1

    bow = dict(freq)
    coherence_score = 0
    print(bow)
    for key, _ in bow.items():
        if bow[key] > 1:
            coherence_score += 1
        else:
            coherence_score += 0

    score = coherence_score / len(bow)
    d["score"] = score

    return d


def process_dict(list_of_wikihow_instances, json_to_write_filename):
    collection = []
    index_for_source = 0
    index_for_target = 1
    for c, wikihow_instance in enumerate(list_of_wikihow_instances, 1):
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
        source_row["ID"] = index_for_source
        source_context = source_row["Context"] = process_context(
            wikihow_instance["Source_Context_5"], wikihow_instance["Source_Line"])

        marked_source, counter = mark_cases(
            source_context, wikihow_instance["PPDB_Matches"], source=True)
        print(marked_source)
        print(counter)
        print("===============")


def main():
    train, test, dev = read_data()

    # process test set
    process_dict(test[0:20],
                 "test_set_pytorch_discourse_sim_ttr_length_right_sim_test.json")


main()
