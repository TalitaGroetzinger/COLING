import bz2
import os
import pickle
import json
from progress.bar import Bar


def get_line_and_file(filename, line_nr, collection):
    # get current_line
    current_line = collection[filename][line_nr]
    # make a list of all line numbers
    sentence_nrs = [key for key, _ in collection[filename].items()]
    list_positions = [i for i in range(len(sentence_nrs))]
    # get the index of the line before and after
    window_range = [1, 2, 3, 4, 5]
    sents_before_current = []
    sents_after_current = []
    for window in window_range:
        previous_line_index = sentence_nrs.index(line_nr)-window
        next_line_index = sentence_nrs.index(line_nr)+window
        if previous_line_index in list_positions:
            previous_line_pos = sentence_nrs[previous_line_index]
            previous_line = collection[filename][previous_line_pos]
            sents_before_current.append(previous_line)

        if next_line_index in list_positions:
            next_line_pos = sentence_nrs[next_line_index]
            next_line = collection[filename][next_line_pos]
            sents_after_current.append(next_line)
    sents_before_current.reverse()
    full_context = sents_before_current + [current_line] + sents_after_current
    return full_context


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
        bar = Bar('Processing', max=len(content))
        for wikihow_instance in content:
            bar.next()
            if wikihow_instance['Loc_in_splits'] == 'DEV':
                source_line_nr = wikihow_instance['Source_Line_Nr'][0]
                target_line_nr = wikihow_instance['Target_Line_Nr'][-1]
                filename = wikihow_instance['Filename']
                filename_key = filename + '.bz2'
                source_line_nr_content = get_line_and_file(
                    filename_key, source_line_nr, collection)
                wikihow_instance['Source_Context'] = source_line_nr_content
                target_line_nr_content = get_line_and_file(
                    filename_key, target_line_nr, collection)
                wikihow_instance['Target_Context'] = target_line_nr_content
                new.append(wikihow_instance)
            bar.finish()

    with open('DIFF-NOUN-MODIFICATIONS-LINE-NR.json', 'w') as json_out:
        json.dump(new, json_out)


main()
