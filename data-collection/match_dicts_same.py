import json
import nltk
from nltk import word_tokenize
from progress.bar import Bar
from collections import Counter
import pickle


def convert_dict(wikihow_collection):
    collection = {}
    double_cases = []
    for wikihow_instance in wikihow_collection:
        new_row = {}
        source_line = wikihow_instance['All_Versions'][0]
        target_line = wikihow_instance['All_Versions'][-1]
        key = wikihow_instance['key']
        if source_line not in collection.keys():
            collection[key] = new_row
            new_row['Filename'] = wikihow_instance['Filename']
            new_row['Revision_Length'] = wikihow_instance['Revision_Length']
            new_row['key'] = wikihow_instance['key']
            new_row['Target_Line'] = target_line
            new_row['Source_Line'] = source_line
            new_row['Source_Line_Nr'] = wikihow_instance['Source_Line_Nr']
            new_row['Target_Line_Nr'] = wikihow_instance['Target_Line_Nr']
        else:
            double_cases.append(wikihow_instance)
    return collection


def flatten_second_dict(json_in):
    collection = {}
    for wikihow_instance in json_in:
        collection[wikihow_instance['Key']] = wikihow_instance
    return collection


def main():
    with open('potential-same-noun-cases-ALL.json', 'r') as json_in:
        potential_diff_noun_cases = json.load(json_in)

    with open('../classification-scripts/same-noun-modifications/same-noun-modifications-keys.json', 'r') as json_in2:
        diff_noun_modifications = json.load(json_in2)

    correct_dict = flatten_second_dict(diff_noun_modifications)
    potential_dict = convert_dict(potential_diff_noun_cases)

    keys_not_found = []
    data = []
    for key, _ in correct_dict.items():
        try:
            # get info from other dict
            target_line_nr = potential_dict[key]['Target_Line_Nr']
            source_line_nr = potential_dict[key]['Source_Line_Nr']

            # update the current dict
            correct_dict[key]['Target_Line_Nr'] = target_line_nr
            correct_dict[key]['Source_Line_Nr'] = source_line_nr
            data.append(correct_dict[key])
        except KeyError:
            print("not found ...")

    with open('../data/SAME-NOUN-MODIFICATIONS-LINE-NR.json', 'w') as json_out:
        json.dump(data, json_out)


main()
