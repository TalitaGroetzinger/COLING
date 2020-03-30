# This script can be used to add the key ['Loc_in_splits'] to a collection of Wikihowinstances.


from collections import Counter
import json
from progress.bar import Bar


def read_list_of_filenames(path_to_file):
    list_of_files = []
    with open(path_to_file, 'r') as file:
        content = file.readlines()
        for line in content:
            line = line.strip('\n')
            list_of_files.append(line)
    assert len(list_of_files) == len(content)
    return list_of_files


def check_filenames_in_json(json_file, list_of_dev_files, list_of_test_files):
    collection = []
    counter = Counter()
    bar = Bar('Processing ', max=len(json_file))
    for wikihow_instance in json_file:
        bar.next()
        if wikihow_instance['Filename'] in list_of_dev_files:
            wikihow_instance['Loc_in_splits'] = 'DEV'
        elif wikihow_instance['Filename'] in list_of_test_files:
            wikihow_instance['Loc_in_splits'] = 'TEST'
        else:
            wikihow_instance['Loc_in_splits'] = 'TRAIN'
        collection.append(wikihow_instance)
        counter[wikihow_instance['Loc_in_splits']] += 1
    #del wikihow_instance['All_Versions']
    bar.finish()
    try:
        assert len(collection) == len(json_file)
    except AssertionError:
        print("the length is not equal: Length of filtered {0}, Length of original {1}".format(
            len(collection), len(json_file)))
    return collection


def main():
    list_of_dev_files = read_list_of_filenames(
        '../classification-scripts/scripts_from_server/wikihow_dev_files.txt')
    list_of_test_files = read_list_of_filenames(
        '../classification-scripts/scripts_from_server/wikihow_test_files.txt')

    with open('../wikihowtools/data/Wikihow_tokenized_v5_cleaned.json', 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)
    print(len(list_of_wikihow_instances))

    collection = check_filenames_in_json(
        list_of_wikihow_instances, list_of_dev_files, list_of_test_files)

    with open('./Wikihow_tokenized_v5_cleaned_splits.json', 'w') as json_out:
        json.dump(collection, json_out)


main()
