import json


def get_paths(different_nouns=True):
    if different_nouns:
        path_to_train = '../classification-scripts/different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TRAIN-5-v3.JSON'
        path_to_dev = '../classification-scripts/different-noun-modifications/DIFF-NOUN-MODIFICATIONS-DEV-5-v3.JSON'
        path_to_test = '../classification-scripts/different-noun-modifications/DIFF-NOUN-MODIFICATIONS-TEST-5-v3.JSON'
    else:
        path_to_train = '../classification-scripts/same-noun-modifications/SAME-NOUN-MODIFICATIONS-TRAIN-5-v3.JSON'
        path_to_dev = '../classification-scripts/same-noun-modifications/SAME-NOUN-MODIFICATIONS-DEV-5-v3.JSON'
        path_to_test = '../classification-scripts/same-noun-modifications/SAME-NOUN-MODIFICATIONS-TEST-5-v3.JSON'
    return path_to_train, path_to_dev, path_to_test


def write_to_json(file_name, list_to_dump):
    with open(file_name, 'w') as json_out:
        json.dump(list_to_dump, json_out)

# different-noun modifications -> wikihow_instance['Source_tagged']
# same-noun modificatioss -> 'Source_Line_Tagged'


def rename_keys(list_of_wikihow_instances):
    renamed_instance = []
    for wikihow_instance in list_of_wikihow_instances:
        wikihow_instance['Source_Line_Tagged'] = wikihow_instance['Source_tagged']
        wikihow_instance['Target_Line_Tagged'] = wikihow_instance['Target_Tagged']
        del wikihow_instance['Source_tagged']
        del wikihow_instance['Target_Tagged']
        renamed_instance.append(wikihow_instance)
    return renamed_instance


def merge_lists(list_with_diff_nouns, list_with_same_nouns):
    with open(list_with_diff_nouns, 'r') as json_in1:
        diff_nouns = json.load(json_in1)

    with open(list_with_same_nouns, 'r') as json_in2:
        same_nouns = json.load(json_in2)
    print(len(diff_nouns+same_nouns))

    diff_nouns_new = rename_keys(diff_nouns)

    return diff_nouns_new + same_nouns


def main():
    train_diff, dev_diff, test_diff = get_paths(True)

    train_same, dev_same, test_same = get_paths(False)

    all_train = merge_lists(train_diff, train_same)
    all_dev = merge_lists(dev_diff, dev_same)
    all_test = merge_lists(test_diff, test_same)

    path_to_dir = '../classification-scripts/noun-modifications'
    print("merge train ... ")
    write_to_json(
        "{0}/noun-modifications-train-5-v3.json".format(path_to_dir), all_train)
    print("Merge dev ..")
    write_to_json(
        "{0}/noun-modifications-dev-5-v3.json".format(path_to_dir), all_dev)
    print("Merge test")
    write_to_json(
        "{0}/noun-modifications-test-5-v3.json".format(path_to_dir), all_test)


main()
