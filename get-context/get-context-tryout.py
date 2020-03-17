import bz2
import os
import pickle
import json


def get_line_and_file(filename, line_nr, collection):
    current_line = collection[filename][line_nr]
    return current_line


def files_to_dict():
    collection = {}
    for filename in os.listdir('dev-files-example'):
        path = "dev-files-example/{0}".format(filename)
        with bz2.open(path, "rt") as bz_file:
            file_in_dict_format = {}
            for counter, line in enumerate(bz_file, 1):
                if line != '\n':
                    file_in_dict_format[counter] = line.strip('\n').strip()
        collection[filename] = file_in_dict_format
    return collection


def main():
    print("load pickle")
    with open('dev-files-in-dict-format.pickle', 'rb') as pickle_in:
        collection = pickle.load(pickle_in)

    print("load json")
    with open('../classification-scripts/classification-data/DIFF-NOUN-MODIFICATIONS-LINE-NR.json', 'r') as json_in:
        content = json.load(json_in)

        new = []
        for wikihow_instance in content:
            if wikihow_instance['Loc_in_splits'] == 'DEV':
                source_line_nr = wikihow_instance['Source_Line_Nr'][0]
                target_line_nr = wikihow_instance['Target_Line_Nr'][-1]
                filename = wikihow_instance['Filename']
                filename_key = filename + '.bz2'
                source_line_nr_content = get_line_and_file(
                    filename_key, source_line_nr, collection)
                wikihow_instance['found_source_line'] = source_line_nr_content
                new.append(wikihow_instance)

    for elem in new[0:10]:
        print(elem['Source_tagged'])
        print(elem['found_source_line'])
    print("-----------------")


main()
