from collections import Counter
import pickle
from nltk.tokenize import word_tokenize
import nltk
from wikihowtools.add_linguistic_info import read_json, compute_char_distance
from pprint import pprint


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
        for key, _ in counts.items():
            if key == 'NN':
                nn_counts += counts[key]
            if key == 'NNS':
                nns_counts += counts[key]
            if key == 'NNP':
                nnp_counts += counts[key]
    d = {"NN": nn_counts, "NNS": nns_counts, "NNP": nnp_counts,
         "total": nn_counts + nns_counts + nnp_counts}
    print(d)


def count_rev_length(corrections):
    counter_dict = Counter()
    rev_lengths = [wikihow_instance["Revision_Length"]
                   for wikihow_instance in corrections if wikihow_instance['Revision_Length'] > 1]
    for c in rev_lengths:
        counter_dict["Revision Length " + str(c)] += 1
    print(dict(counter_dict))


def tag_sent(wikihow_instance):
    """
      POS-tag all the cases in wikihow_instance["All_Versions"]
    """
    wikihow_instance_all_versions = wikihow_instance["All_Versions"]
    tagged_sents = []
    for sent in wikihow_instance_all_versions:
        tokenized = word_tokenize(sent)
        tagged = nltk.pos_tag(tokenized)
        tagged_sents.append(tagged)
    wikihow_instance["All_tagged"] = tagged_sents
    return wikihow_instance


def main():
    corrections = pickle.load(open("./data/real_corrections.pickle", "rb"))
    noun_corrections = get_difference_by_tags(corrections)
    tag_intermediate_revisions(noun_corrections)

    # with open('./data/real_corrections_nouns.pickle', 'wb') as pickle_out:
    #    pickle.dump(noun_corrections, pickle_out)


main()
