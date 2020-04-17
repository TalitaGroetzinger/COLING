# This script was use do add ['Source_Line'] and ['Target_Line'] wikihow_v5_tokenized.json

import json
from progress.bar import Bar
import argparse


def flatten_dict(json_in):
    # convert dict in such a format that the "Key" will be the key
    # I need this for wikihow_v5_tokenized_lines.json
    collection = {}
    for wikihow_instance in json_in:
        collection[wikihow_instance['Key']] = wikihow_instance
    return collection


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Train and test classifier")
    ap.add_argument("--input", required=True, type=str,
                    help="File to read")
    ap.add_argument("--output", required=True, type=str,
                    help="File to write")

    args = vars(ap.parse_args())
    path_to_nouns = args['input']
    filename_to_write = args['output']

    print("read wikihow_tokenized_v5_lines.json")
    with open('wikihow_tokenized_v5_lines.json', 'r') as json_in_all:
        wikihow_all = json.load(json_in_all)

    print("read noun modifications")
    with open(path_to_nouns, 'r') as json_in:
        wikihow_noun_mod = json.load(json_in)

    # make dict for complete data
    complete_data = flatten_dict(wikihow_all)

    new_instances = []
    bar = Bar("Processing ... ", max=len(wikihow_noun_mod))
    for wikihow_instance in wikihow_noun_mod:
        bar.next()
        key_to_find = wikihow_instance['Key']
        try:
            source_line = complete_data[key_to_find]['Source_Line']
            target_line = complete_data[key_to_find]['Target_Line']
            wikihow_instance['Source_Line'] = source_line
            wikihow_instance['Target_Line'] = target_line
        except KeyError:
            print("Key {0} not found".format(key_to_find))
        new_instances.append(wikihow_instance)
    bar.finish()

    assert len(new_instances) == len(wikihow_noun_mod)

    print("write file with name {0}".format(filename_to_write))
    with open(filename_to_write, 'w') as json_out:
        json.dump(new_instances, json_out)
