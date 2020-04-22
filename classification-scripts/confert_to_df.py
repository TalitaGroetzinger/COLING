# this script can be used to fit different columns to the vectorizer
import json
from collections import Counter


def get_paths():
    path_to_dir = './noun-modifications'
    path_to_test = '{0}/noun-modifications-test-v2-new.json'.format(
        path_to_dir)
    path_to_train = '{0}/noun-modifications-train-v2-new.json'.format(
        path_to_dir)
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(path_to_dir)
    return path_to_train, path_to_dev, path_to_test


def main():
    path_to_train, path_to_dev, path_to_test = get_paths()
    with open(path_to_dev, 'r') as json_in:
        data = json.load(json_in)
    print(len(data))

    subset = data[0:10]
    print(subset[1]['Source_Line'])


main()
