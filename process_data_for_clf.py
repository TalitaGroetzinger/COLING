# This script can be used to add any additional information about the PPDB info to the
# wikihow_instance, f.i. because this might be relevant for classification purposes.

from parse_db import read_ppdb
from progress.bar import Bar
from wikihowtools.add_linguistic_info import read_json
import json


def add_entailment_relations(list_of_wikihow_instances, pdb):
    collection = []
    bar = Bar('Processing', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        entailment_relations = {}
        for pair in wikihow_instance['Differences']:
            source = pair[0]
            target = pair[1]
            key_to_look_for = "{0}#{1}".format(
                source[0].lower(), target[0].lower())
            try:
                entailment_relations[key_to_look_for] = pdb[key_to_look_for]['ENTAILMENT']
            except KeyError:
                continue
            if entailment_relations != {}:
                wikihow_instance["Entailment_Rel"] = entailment_relations
                collection.append(wikihow_instance)
        bar.next()
    bar.finish()

    return collection


def main():
    content = read_json('./data/noun_corrections.json')
    ppdb = read_ppdb('./data/ppdb/ppdb-xxxl-lexical.txt')
    list_with_pdb_relations = add_entailment_relations(
        content, ppdb)
    print(len(list_with_pdb_relations))
    with open('noun_corrections_ppdb_tagged.json', 'w') as json_file:
        json.dump(list_with_pdb_relations, json_file)


main()
