from wikihowtools.add_linguistic_info import read_json, compute_char_distance
import pickle


def get_corrections_edit_distance(potential_corrections, edit_distance_value=2):
    real_corrections = []
    for wikihow_instance in potential_corrections:
        diff_words_per_instance = []
        for i in range(len(wikihow_instance['Source_tagged'])):
            # go through each source_tagged and target_tagged in parallel -> possible to do this because of equal length
            # not necessary here to do source_tagged[i][1] because the tags will never differ.
            if not wikihow_instance['Source_tagged'][i] == wikihow_instance['Target_Tagged'][i]:
                # compute the distance of there is a difference between
                distance = compute_char_distance(
                    wikihow_instance['Source_tagged'][i][0], wikihow_instance['Target_Tagged'][i][0])
                if distance > edit_distance_value:
                    diff_words = [wikihow_instance['Source_tagged'][i],
                                  wikihow_instance['Target_Tagged'][i], distance]
                    diff_words_per_instance.append(diff_words)
                    wikihow_instance['differences'] = diff_words_per_instance
        del wikihow_instance['Revisions']
        del wikihow_instance['Source_Tokenized']
        del wikihow_instance['Target_Tokenized']
        del wikihow_instance['Correction']
        del wikihow_instance['Key']
        del wikihow_instance['Base_Sentence']
        if "differences" in wikihow_instance.keys():
            real_corrections.append(wikihow_instance)
    return real_corrections


def main():
    # read the cases which contain the same length + pos tags
    corrections = read_json(
        './wikihowtools/data/wikihow_tokenized_tagged_possible_corrections.json')

    print("get real corrections")
    real_corrections = get_corrections_edit_distance(corrections)
    assert len(real_corrections) == 124685
    print(real_corrections[0].keys())
    print(real_corrections[0])
    with open('./data/real_corrections.pickle', 'wb') as pickle_out:
        pickle.dump(real_corrections, pickle_out)


main()
