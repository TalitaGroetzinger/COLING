# use this script to use wikihowinstance['loc_in_splits']

from progress.bar import Bar


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
            wikihow_instance['Loc_in_splits'] == 'TRAIN':
            train.append(wikihow_instance)

    return dev, test, train


def main():
    path = '../wikihowtools/data/Wikihow_tokenized_v5_cleaned_tokens_only.json'
    with open(path, 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)
    dev, test, train = make_splits(list_of_wikihow_instances)

    print("Train instances: {0}, Perc: {1}".len(train),
          len(train)/len(list_of_wikihow_instances))
    print("Dev instances: {0}, Perc: {1}".len(dev), len(
        dev)/len(list_of_wikihow_instances))
    print("Test instances: {0}, Perc: {1}".len(test), len(
        test)/len(list_of_wikihow_instances))

    with open("wikihow-dev.json", 'w') as json_out:
        json.dump(json_out, dev)
    with open("wikihow-test.json", 'w') as json_out:
        json.dump(json_out, test)
    with open("wikihow-train.json", 'w') as json_out:
        json.dump(json_out, train)


main()
