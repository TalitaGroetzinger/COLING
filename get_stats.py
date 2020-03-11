# Use this script to get the frequency distribution of nouns changing into other nouns

import json
from collections import Counter


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


def main():
    with open('./noun_corrections_INC_ED.json', 'r') as json_file:
        noun_corrections = json.load(json_file)
    res, total = count_tags(noun_corrections)
    print(res)
    print(total)


main()
