# use this script to use wikihowinstance['loc_in_splits']

from progress.bar import Bar
import json


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
    return dev, test, train


def main():
    path = '../wikihowtools/data/Wikihow_tokenized_v5_cleaned_splits_tokens_only.json'
    with open(path, 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)

    print(len(list_of_wikihow_instances))

    print("make the splits ... ")
    dev, test, train = make_splits(list_of_wikihow_instances)

    print("Train instances: {0}, Perc: {1}".format(len(train),
                                                   len(train)/len(list_of_wikihow_instances)))
    print("Dev instances: {0}, Perc: {1}".format(len(dev), len(
        dev)/len(list_of_wikihow_instances)))
    print("Test instances: {0}, Perc: {1}".format(len(test), len(
        test)/len(list_of_wikihow_instances)))

    with open("wikihow-dev.json", 'w') as json_out:
        json.dump(dev, json_out)
    with open("wikihow-test.json", 'w') as json_out:
        json.dump(test, json_out)
    with open("wikihow-train.json", 'w') as json_out:
        json.dump(train, json_out)


main()
