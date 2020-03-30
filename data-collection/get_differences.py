
# This script was used to get all the corrections (and to not exclude cases with edit distance < 2)

import json


def get_differences(corrections):
    collection = []
    for wikihow_instance in corrections:
        differences = []
        source_tagged = wikihow_instance['Source_tagged']
        target_tagged = wikihow_instance['Target_Tagged']
        for source, target in zip(source_tagged, target_tagged):
            if source[0] != target[0]:
                differences.append([source, target])
        wikihow_instance['differences'] = differences
        del wikihow_instance['Key']
        del wikihow_instance['All_Versions']
        del wikihow_instance['Source_Tokenized']
        del wikihow_instance['Target_Tokenized']
        del wikihow_instance['Base_Sentence']
        del wikihow_instance['Correction']
        del wikihow_instance['Revisions']
        if differences != []:
            collection.append(wikihow_instance)
    return collection


def main():
    with open('./data/wikihow_tokenized_tagged_possible_corrections.json', 'r') as json_in:
        corrections = json.load(json_in)

    new_corrections = get_differences(corrections)
    print(len(new_corrections))
    with open('./data/wikihow_same_pos_same_length_NEW.json', 'w') as json_out:
        json.dump(new_corrections, json_out)


main()
