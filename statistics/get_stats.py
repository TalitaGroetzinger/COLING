# Use this script to get the frequency distribution of nouns changing into other nouns

import json
from collections import Counter
import nltk


def count_tags(list_of_wikihow_instances):
    all_tags = []
    for wikihow_instance in list_of_wikihow_instances:
        for elem in wikihow_instance['Differences']:
            source = elem[0]
            source_tag = source[1]
            all_tags.append(source_tag)
    freq_dist = Counter()
    for tag in all_tags:
        freq_dist[tag] += 1
    return freq_dist, len(all_tags)


def count_tags_diff_nouns(list_of_wikihow_instances):
    freq_dist_PPDB = Counter()
    modifications = []
    all_differences = []
    freq_dist_all = Counter()
    for wikihow_instance in list_of_wikihow_instances:
        ppdb_matches = wikihow_instance['PPDB_Matches']
        differences = wikihow_instance['Differences']
        for pair in ppdb_matches:
            source_tag = pair[0][1]
            target_tag = pair[1][1]
            modification = "{0}#{1}".format(source_tag, target_tag)
            modifications.append(modification)
        for difference in differences:
            source_tag = pair[0][1]
            target_tag = pair[1][1]
            modification_in_differences = "{0}#{1}".format(
                source_tag, target_tag)
            all_differences.append(modification_in_differences)

    for modification in modifications:
        freq_dist_PPDB[modification] += 1
    for difference in all_differences:
        freq_dist_all[difference] += 1

    print("PPDB STATS")
    print(len(list_of_wikihow_instances))
    print(len(modifications))
    print(freq_dist_PPDB)
    print("GENERAL")
    print(freq_dist_all)


def main():
    """
    with open('./noun_corrections_ppdb_tagged_v3_with_split_info.json', 'r') as json_file:
        noun_corrections = json.load(json_file)
    res, total = count_tags(noun_corrections)
    print(res)
    print(total)
    count_rels(noun_corrections)
    """
    with open('../data/diff_noun_modifications_PPDB_tagged.json', 'r') as json_in:
        content = json.load(json_in)

    # count_tags_diff_nouns(content)

    """
    for wikihow_instance in content:
        for elem in wikihow_instance['PPDB_Matches']:
            source_tag = elem[0][1]
            target_tag = elem[1][1]
            if source_tag == 'NNS' and target_tag == 'NNP':
                print(elem)
    """
    for wikihow_instance in content:
        for key, _ in wikihow_instance['Entailment_Rel'].items():
            if wikihow_instance['Entailment_Rel'][key] == 'Exclusion':
                print(wikihow_instance['Entailment_Rel'])


main()
