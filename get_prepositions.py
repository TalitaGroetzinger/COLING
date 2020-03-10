from wikihowtools.add_linguistic_info import read_json
import json
from progress.bar import Bar
import spacy

spacy_tagger = spacy.load('en_core_web_sm')


def get_pps_from_sent(sent, token_to_look_for='ADP'):
    """
      Input: A string
      Output: A list of preposition phrases occuring in that string.
    """
    parsed_sent = spacy_tagger(sent)
    pps = []
    for token in parsed_sent:
        if token.pos_ == token_to_look_for:
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
    return pps


def check_difs(list_of_wikihow_instance):
    collection = []
    bar = Bar('Processing', max=len(list_of_wikihow_instance))
    for wikihow_instance in list_of_wikihow_instance:
        diff_preps = []
        source_tokens = " ".join(wikihow_instance['Source_Tokenized'])
        target_tokens = " ".join(wikihow_instance['Target_Tokenized'])
        PP_in_source = get_pps_from_sent(source_tokens)
        PP_in_target = get_pps_from_sent(target_tokens)
        source_pps_set = set(PP_in_source)
        non_prepositions = ["if", "that"]
        for token in PP_in_target:
            if token not in source_pps_set:
                if token.lower() not in non_prepositions:
                    diff_preps.append(token)
                    wikihow_instance['Possible_PPS_Insertions'] = diff_preps
        if diff_preps != []:
            del wikihow_instance['All_Versions']
            del wikihow_instance['Base_Sentence']
            del wikihow_instance['Revisions']
            del wikihow_instance['Source_tagged']
            del wikihow_instance['Target_Tagged']
            del wikihow_instance['Key']
            collection.append(wikihow_instance)
        bar.next()
    bar.finish()
    return collection


def main():
    insertions = read_json('./data/insertions.json')
    new_insertions = check_difs(insertions)
    with open('pp_insertions_new.json', 'w') as json_out:
        json.dump(new_insertions, json_out)
    print("length of insertions", len(insertions))
    print("Possible pp-insertions", len(new_insertions))


main()
