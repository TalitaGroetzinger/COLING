from collections import Counter
import pickle


def get_difference_by_tags(corrections, list_of_tags=["NN", "NNS", "NNP"]):
    filtered = []
    for wikihow_instance in corrections:
        differences = wikihow_instance['differences']
        list_of_changed_tags = [difference[0][1] for difference in differences]
        assert len(list_of_changed_tags) == len(differences)
        tags_counter = Counter()
        for tag in list_of_changed_tags:
            if tag in list_of_tags:
                tags_counter[tag] += 1
        if tags_counter != {}:
            wikihow_instance["Tags_Count"] = dict(tags_counter)
            filtered.append(wikihow_instance)
    return filtered


def get_freq_dist(filtered_corrections):
    nn_counts = 0
    nns_counts = 0
    nnp_counts = 0
    for wikihow_instance in filtered_corrections:
        counts = wikihow_instance["Tags_Count"]
        for key, value in counts.items():
            if key == 'NN':
                nn_counts += counts[key]
            if key == 'NNS':
                nns_counts += counts[key]
            if key == 'NNP':
                nnp_counts += counts[key]
    d = {"NN": nn_counts, "NNS": nns_counts, "NNP": nnp_counts,
         "total": nn_counts + nns_counts + nnp_counts}
    print(d)


def main():
    corrections = pickle.load(open("./data/real_corrections.pickle", "rb"))
    noun_corrections = get_difference_by_tags(corrections)
    counter = 0
    for elem in noun_corrections:
        if elem['Revision_Length'] > 1:
            counter += 1

    print(counter)
    get_freq_dist(noun_corrections)

    # with open('./data/real_corrections_nouns.pickle', 'wb') as pickle_out:
    #    pickle.dump(noun_corrections, pickle_out)


main()
