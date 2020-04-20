import bz2
import os
import pickle
import json


def get_file_to_dict_format(directory='dev-files'):
    data = {}
    for filename in os.listdir(directory):
        path = "./{0}/{1}".format(directory, filename)
        with bz2.open(path, "rt") as bz_file:
            file_in_dict_format = {}
            for counter, line in enumerate(bz_file, 1):
                if line != '\n':
                    file_in_dict_format[counter] = line.strip('\n').strip()
        data[filename] = file_in_dict_format
    return data


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
    #full_context = sents_before_current + [current_line] + sents_after_current
    # return full_context
    return {'left': sents_before_current,
            'current': current_line, 'right': sents_after_current}


def main():
    print("Get files ...")
    files_in_dict_format = get_file_to_dict_format()

    with open('train-dict-format.json', 'w') as json_out:
        json.dump(files_in_dict_format, json_out)


main()
