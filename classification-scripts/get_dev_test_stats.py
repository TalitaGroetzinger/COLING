# this script can be used to add the key ['Loc_in_splits'] to a collection of Wikihowinstances.


from collections import Counter
import json


def json_to_txt(json_file):
    collection = []
    for wikihow_instance in json_file:
        if wikihow_instance['Loc_in_splits'] == 'TRAIN':
            collection.append(wikihow_instance['Filename'])
    return collection


def get_list_of_filenames(path_to_file):
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
    for wikihow_instance in json_file:
        if wikihow_instance['Filename'] in list_of_dev_files:
            wikihow_instance['Loc_in_splits'] = 'DEV'
        elif wikihow_instance['Filename'] in list_of_test_files:
            wikihow_instance['Loc_in_splits'] = 'TEST'
        else:
            wikihow_instance['Loc_in_splits'] = 'TRAIN'
        collection.append(wikihow_instance)
        counter[wikihow_instance['Loc_in_splits']] += 1
    #del wikihow_instance['All_Versions']
    print(counter)
    print(len(collection))
    return collection


def main():
    list_of_dev_files = get_list_of_filenames(
        './classification-data/dev_files.txt')
    list_of_test_files = get_list_of_filenames(
        './classification-data/test_files.txt')

    with open('./classification-data/noun_corrections_ppdb_tagged_v3_with_split_info.json', 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)
    print(len(list_of_wikihow_instances))

    collection = check_filenames_in_json(
        list_of_wikihow_instances, list_of_dev_files, list_of_test_files)

    # to get train files:
    with open('../classification-scripts/classification-data/DIFF-NOUN-MODIFICATIONS.json') as json_in:
        content = json.load(json_in)

    collection = json_to_txt(content)
    with open('train.txt', 'w') as file_out:
        for filename in collection:
            line_to_write = "{0}\n".format(filename)
            file_out.write(line_to_write)


main()
