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
    # del wikihow_instance['All_Versions']
    bar.finish()
    try:
        assert len(collection) == len(json_file)
    except AssertionError:
        print("the length is not equal: Length of filtered {0}, Length of original {1}".format(
            len(collection), len(json_file)))
    return collection


def make_splits(list_of_wikihow_instances):
    bar = Bar('Processing ', max=len(list_of_wikihow_instances))
    dev = []
    test = []
    train = []
    for wikihow_instance in list_of_wikihow_instances:
        bar.next()
        if wikihow_instance['Loc_in_splits'] == 'DEV':
            dev.append(wikihow_instance)
        elif wikihow_instance['Loc_in_splits'] == 'TEST':
            test.append(wikihow_instance)
        else:
            train.append(wikihow_instance)
    bar.finish()
    return train, dev, test


def main():
    list_of_dev_files = read_list_of_filenames(
        '../classification-scripts/scripts_from_server/wikihow_dev_files.txt')
    list_of_test_files = read_list_of_filenames(
        '../classification-scripts/scripts_from_server/wikihow_test_files.txt')

    with open('../wikihowtools/data/Wikihow_tokenized_v5_cleaned_tokens_only.json', 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)

    print("add info in wikihow_instance ... ")
    list_of_wikihow_instances_new = check_filenames_in_json(
        list_of_wikihow_instances, list_of_dev_files, list_of_test_files)

    print("split into different files .... ")
    dev, test, train = make_splits(list_of_wikihow_instances_new)

    print("Train instances: {0}, Perc: {1}".format(len(train),
                                                   len(train)/len(list_of_wikihow_instances_new)))
    print("Dev instances: {0}, Perc: {1}".format(len(dev), len(
        dev)/len(list_of_wikihow_instances_new)))
    print("Test instances: {0}, Perc: {1}".format(len(test), len(
        test)/len(list_of_wikihow_instances_new)))

    with open("wikihow-dev.json", 'w') as json_out:
        json.dump(json_out, dev)
    with open("wikihow-test.json", 'w') as json_out:
        json.dump(json_out, test)
    with open("wikihow-train.json", 'w') as json_out:
        json.dump(json_out, train)


main()
