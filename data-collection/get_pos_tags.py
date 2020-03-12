from collections import Counter
import pickle
from nltk.tokenize import word_tokenize
import nltk
from pprint import pprint
import json
from progress.bar import Bar


def count_rev_length(corrections):
    counter_dict = Counter()
    rev_lengths = [wikihow_instance["Revision_Length"]
                   for wikihow_instance in corrections if wikihow_instance['Revision_Length'] > 1]
    for c in rev_lengths:
        counter_dict["Revision Length " + str(c)] += 1
    print(dict(counter_dict))


def filter_insertions(wikihow_instance_collection):
    collection = []
    for wikihow_instance in wikihow_instance_collection:
        filtered_differences = []
        for elem in wikihow_instance['differences']:
            source = elem[0]
            tag_source = source[1]
            if 'NN' == tag_source or 'NNS' == tag_source or 'NNP' == tag_source:
                filtered_differences.append(elem)
        wikihow_instance['Source_Line_Tagged'] = wikihow_instance['Source_tagged']
        wikihow_instance['Target_Line_Tagged'] = wikihow_instance['Target_Tagged']
        wikihow_instance['Differences'] = filtered_differences
        del wikihow_instance['Source_tagged']
        del wikihow_instance['Target_Tagged']
        del wikihow_instance['differences']
        if wikihow_instance['Differences'] != []:
            collection.append(wikihow_instance)
    return collection


def get_different_noun_modifications(wikihow_instance_collection):
    collection = []
    bar = Bar('Processing', max=len(wikihow_instance_collection))
    for wikihow_instance in wikihow_instance_collection:
        list_of_differences = []
        list_of_noun_modifications = []
        source_tagged = wikihow_instance['Source_tagged']
        target_tagged = wikihow_instance['Target_Tagged']
        for source_word_tag_pair, target_word_tag_pair in zip(source_tagged, target_tagged):
            if source_word_tag_pair[1] != target_word_tag_pair[1]:
                list_of_differences.append(
                    [source_word_tag_pair, target_word_tag_pair])
        for difference in list_of_differences:
            source_tag = difference[0][1]
            target_tag = difference[1][1]
            if source_tag == 'NN' and (target_tag == 'NNP' or target_tag == 'NNS'):
                list_of_noun_modifications.append(difference)
            elif source_tag == 'NNS' and (target_tag == 'NN' or target_tag == 'NNP'):
                list_of_noun_modifications.append(difference)
            elif source_tag == 'NNP' and (target_tag == 'NNS' or target_tag == 'NN'):
                list_of_noun_modifications.append(difference)
            else:
                continue
        bar.next()
        if list_of_noun_modifications != []:
            wikihow_instance['Differences'] = list_of_noun_modifications
            del wikihow_instance['Base_Sentence']
            del wikihow_instance['Revisions']
            del wikihow_instance['Source_Tokenized']
            del wikihow_instance['Target_Tokenized']
            del wikihow_instance['Correction_type2']
            collection.append(wikihow_instance)
    print(len(wikihow_instance_collection))
    print(len(collection))
    bar.finish()
    return collection


def main():
    # corrections = pickle.load(open("./data/real_corrections.pickle", "rb"))
    # with open('./data/wikihow_same_pos_same_length_NEW.json', 'r') as json_in:
    #    corrections = json.load(json_in)

    #noun_corrections = filter_insertions(corrections)
    # with open('noun_corrections_INC_ED.json', 'w') as json_file:
    #    json.dump(noun_corrections, json_file)

    with open('../data/wikihow_tokenized_tagged_possible_corrections_v2.json', 'r') as json_in:
        list_of_wikihow_instances = json.load(json_in)

    noun_modifications = get_different_noun_modifications(
        list_of_wikihow_instances)
    print("----------------------------")
    for elem in noun_modifications[0:10]:
        print(elem['Source_tagged'])
        print(elem['Target_Tagged'])
        print(elem['Differences'])
        print('-----------------------')
    print(len(noun_modifications))

    with open('../data/diff_noun_modifications.json', 'w') as json_in:
        json.dump(noun_modifications, json_in)


main()
