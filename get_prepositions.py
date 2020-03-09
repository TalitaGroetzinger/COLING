from wikihowtools.add_linguistic_info import read_json
import json
from progress.bar import Bar


def diff_tags(list_of_wikihow_instances):
    """
        First argument: target_line
        Second argument: source_line
    """
    possible_pp_insertions = []
    bar = Bar('Processing', max=len(list_of_wikihow_instances))
    for wikihow_instance in list_of_wikihow_instances:
        target = wikihow_instance['Target_Tagged']
        source = wikihow_instance['Source_tagged']
        target_words = [pair[0] for pair in target]
        source_words = [pair[0] for pair in source]
        source_line_set = set(source_words)
        insertions = [[word, tag]
                      for word, tag in target if word not in source_line_set]
        indexes = [target.index([word, tag])
                   for word, tag in target if word not in source_line_set]

        insertion_tags = [word_tag[1] for word_tag in insertions]
        bar.next()
        if 'IN' in insertion_tags:
            wikihow_instance['Insertions'] = insertions
            wikihow_instance['Indexes'] = indexes
            possible_pp_insertions.append(wikihow_instance)
    bar.finish()
    return possible_pp_insertions


def main():
    insertions = read_json('./data/insertions.json')
    print("length of insertions", len(insertions))
    new_insertions = diff_tags(insertions)
    print("Possible pp-insertions", len(new_insertions))
    with open('potential_preposition_insertions.json', 'w') as json_out:
        json.dump(new_insertions, json_out)


main()
