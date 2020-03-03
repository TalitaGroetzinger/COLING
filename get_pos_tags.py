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


def add_differences(wikihow_instance):
    tagged_all = wikihow_instance["All_tagged"]
    collection = []
    for i in range(len(tagged_all)-1):
        current = tagged_all[i]
        next_item = tagged_all[i+1]
        tags_current = [tag[1] for tag in current]
        tags_next = [tag[1] for tag in next_item]
        if tags_current == tags_next:
            d = {"base": current, "target": next_item,
                 "differences": []}
            for current_pair, next_pair in zip(current, next_item):
                if current_pair[0] != next_pair[0]:
                    distance = compute_char_distance(
                        current_pair[0], next_pair[0])
                    if distance > 2:
                        d['differences'].append([current_pair, next_pair])
            collection.append(d)
            wikihow_instance["All_diffs_tagged"] = collection
        else:
            wikihow_instance["All_diffs_tagged"] = False
    return wikihow_instance


def tag_intermediate_revisions(corrections):
    deep_wikihow_cases = [tag_sent(wikihow_instance)
                          for wikihow_instance in corrections if wikihow_instance['Revision_Length'] > 1]
    assert len(deep_wikihow_cases) == 4797
    deep_wikihow_cases_v2 = [add_differences(
        wikihow_instance) for wikihow_instance in deep_wikihow_cases]
    for elem in deep_wikihow_cases_v2:
        if elem['All_diffs_tagged']: 
          pprint(elem["All_Versions"])
          pprint(elem["All_diffs_tagged"])
          print("======================")


def main():
    corrections = pickle.load(open("./data/real_corrections.pickle", "rb"))
    noun_corrections = get_difference_by_tags(corrections)
    tag_intermediate_revisions(noun_corrections)

    # with open('./data/real_corrections_nouns.pickle', 'wb') as pickle_out:
    #    pickle.dump(noun_corrections, pickle_out)


main()
