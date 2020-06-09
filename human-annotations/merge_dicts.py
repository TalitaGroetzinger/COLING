import json
import pickle


def read_data(return_dev=True):
    """
      Function to read the dataset
    """
    path_to_dir = '../classification-scripts/noun-modifications'
    path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(
        path_to_dir)
    if return_dev:
        print("read development set only")
        path_to_dev = '{0}/noun-modifications-dev-v2-new.json'.format(
            path_to_dir)
        with open(path_to_dev, 'r') as json_in_dev:
            dev = json.load(json_in_dev)
        return dev


def make_first_dict(dev):
    index_for_source = 0
    index_for_target = 1
    d = {}
    for c, wikihow_instance in enumerate(dev, 1):
        # process everything for source

        d[index_for_source] = wikihow_instance["Entailment_Rel"]

        index_for_target = index_for_target + 2
        index_for_source = index_for_source + 2

    return d


def main():
    dev_set = read_data()
    print(dev_set[0].keys())
    d = make_first_dict(dev_set)
    for key, value in d.items():
        print(key, '\t',  value)

    # with open('entailent_relations_development_set.pickle', 'wb') as pickle_out:
    #    pickle.dump(d, pickle_out)


main()
