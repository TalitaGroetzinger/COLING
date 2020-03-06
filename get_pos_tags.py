from collections import Counter
import pickle
from nltk.tokenize import word_tokenize
import nltk
from wikihowtools.add_linguistic_info import read_json, compute_char_distance
from pprint import pprint
import json


def get_freq_dist(filtered_corrections):
    nn_counts = 0
    nns_counts = 0
    nnp_counts = 0
    for wikihow_instance in filtered_corrections:
        counts = wikihow_instance["Tags_Count"]
        for key, _ in counts.items():
            if key == 'NN':
                nn_counts += counts[key]
            if key == 'NNS':
                nns_counts += counts[key]
            if key == 'NNP':
                nnp_counts += counts[key]
    d = {"NN": nn_counts, "NNS": nns_counts, "NNP": nnp_counts,
         "total": nn_counts + nns_counts + nnp_counts}
    print(d)


def count_rev_length(corrections):
    counter_dict = Counter()
    rev_lengths = [wikihow_instance["Revision_Length"]
                   for wikihow_instance in corrections if wikihow_instance['Revision_Length'] > 1]
    for c in rev_lengths:
        counter_dict["Revision Length " + str(c)] += 1
    print(dict(counter_dict))


def filter_insertions(wikihow_instance_collection):
    collection = []
    for wikihow_instance in wikihow_instance_collection:
        filtered_differences = []
        for elem in wikihow_instance['differences']:
            source = elem[0]
            tag_source = source[1]
            if 'NN' == tag_source or 'NNS' == tag_source or 'NNP' == tag_source:
                filtered_differences.append(elem)
        wikihow_instance['Source_Line_Tagged'] = wikihow_instance['Source_tagged']
        wikihow_instance['Target_Line_Tagged'] = wikihow_instance['Target_Tagged']
        wikihow_instance['Differences'] = filtered_differences
        del wikihow_instance['Source_tagged']
        del wikihow_instance['Target_Tagged']
        del wikihow_instance['differences']
        if wikihow_instance['Differences'] != []:
            collection.append(wikihow_instance)
    return collection


def main():
    corrections = pickle.load(open("./data/real_corrections.pickle", "rb"))
    noun_corrections = filter_insertions(corrections)
    with open('noun_corrections.json', 'w') as json_file:
        json.dump(noun_corrections, json_file)


main()
