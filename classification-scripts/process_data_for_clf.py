# This script can be used to add any additional information about the PPDB info to the
# wikihow_instance, f.i. because this might be relevant for classification purposes.

from parse_db import read_ppdb
from progress.bar import Bar
import json


def add_entailment_relations(list_of_wikihow_instances, pdb):
    collection = []
    bar = Bar('Processing', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        PPDB_match = []
        entailment_relations = {}
        for counter, pair in enumerate(wikihow_instance['Differences'], 1):
            source = pair[0]
            target = pair[1]
            key_to_look_for = "{0}#{1}".format(
                source[0].lower(), target[0].lower())
            try:
                entailment_relations[key_to_look_for +
                                     str(counter)] = pdb[key_to_look_for]['ENTAILMENT']
                PPDB_match.append(pair)
            except KeyError:
                continue
        if entailment_relations != {}:
            wikihow_instance["Entailment_Rel"] = entailment_relations
            wikihow_instance["PPDB_Matches"] = PPDB_match
            collection.append(wikihow_instance)
        bar.next()
    bar.finish()

    return collection


def main():
    with open('../data/first-step-same-nouns.json', 'r') as json_in:
        content = json.load(json_in)

    ppdb = read_ppdb('../data/ppdb/ppdb-xxxl-lexical.txt')
    list_with_pdb_relations = add_entailment_relations(
        content, ppdb)

    print(len(list_with_pdb_relations))
    with open('../data/second-step-same-nouns.json', 'w') as json_file:
        json.dump(list_with_pdb_relations, json_file)


main()
