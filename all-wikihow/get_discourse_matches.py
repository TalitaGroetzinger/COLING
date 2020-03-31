import json
import pickle
from progress.bar import Bar

# open the markers
markers = '../data/discourse_markers.pickle'
with open(markers, 'rb') as pickle_in:
    markers = pickle.load(pickle_in)


def check_discourse_maches(tokens):
    """
        Input: tokenized sent or document from wikihow_instance
    """
    total = 0
    unigram_matches = 0
    bigram_matches = 0
    trigram_matches = 0
    fourgram_matches = 0
    fivegram_matches = 0
    print("-----")
    # Later zal dit meteen het document zijn/de tokens.
    for token in tokens:
        if token in markers.keys():
            if 'fivegrams' in markers[token].keys():
                fivegrams = [[tokens[i], tokens[i+1], tokens[i+2],
                              tokens[i+3], tokens[i+4]] for i in range(len(tokens)-5)]
                for fivegram in fivegrams:
                    if fivegram in markers[token]['fivegrams']:
                        fivegram_matches += 1
                        total += 1
                        print(fivegram_matches, '#',
                              markers[token]['fivegrams'])
                        print("\n")

            if 'fourgrams' in markers[token].keys():
                fourgrams = [[tokens[i], tokens[i+1], tokens[i+2],
                              tokens[i+3]] for i in range(len(tokens)-4)]
                for fourgram in fourgrams:
                    if fourgram in markers[token]['fourgrams']:
                        fourgram_matches += 1
                        total += 1
                        print(fourgram, '#', markers[token]['fourgrams'])
                        print("\n")

            if 'trigrams' in markers[token].keys():
                trigrams = [[tokens[i], tokens[i+1], tokens[i+2]]
                            for i in range(len(tokens)-3)]
                for trigram in trigrams:
                    if trigram in markers[token]['trigrams']:
                        trigram_matches += 1
                        total += 1
                        print(trigram, '#', markers[token]['trigrams'])
                        print("\n")

            if 'bigrams' in markers[token].keys():
                bigrams = [[tokens[i], tokens[i+1]]
                           for i in range(len(tokens)-2)]
                for bigram in bigrams:
                    if bigram in markers[token]['bigrams']:
                        bigram_matches += 1
                        total += 1
                        print(bigram, markers[token]['bigrams'])
                        print("\n")
            if 'unigrams' in markers[token].keys():
                print(token, '#', markers[token]['unigrams'])
                unigram_matches += 1
                total += 1
    return {"score": unigram_matches + bigram_matches + trigram_matches + fourgram_matches + fivegram_matches}


def get_paths(different_nouns=True):
    path_to_dir_diff = '../classification-scripts/different-noun-modifications/'
    path_to_dir_same = '../classification-scripts/same-noun-modifications/'
    if different_nouns:
        path_to_train = '{0}/DIFF-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'.format(
            path_to_dir_diff)
        path_to_dev = '{0}/DIFF-NOUN-MODIFICATIONS-DEV-5-new.JSON'.format(
            path_to_dir_diff)
        path_to_test = '{0}/DIFF-NOUN-MODIFICATIONS-TEST-5-new.JSON'.format(
            path_to_dir_diff)
    else:
        path_to_train = '{0}/SAME-NOUN-MODIFICATIONS-TRAIN-5-new.JSON'.format(
            path_to_dir_same)
        path_to_dev = '{0}/SAME-NOUN-MODIFICATIONS-DEV-5-new.JSON'.format(
            path_to_dir_same)
        path_to_test = '{0}/SAME-NOUN-MODIFICATIONS-TEST-5-new.JSON'.format(
            path_to_dir_same)
    return path_to_train, path_to_dev, path_to_test


def get_data(path_to_train, path_to_dev, path_to_test, diff_nouns=True, source=True):
    with open(path_to_train, 'r') as json_in:
        train = json.load(json_in)
    with open(path_to_test, 'r') as json_in:
        test = json.load(json_in)
    with open(path_to_dev, 'r') as json_in:
        dev = json.load(json_in)

    all_data = train+dev+test

    documents = []
    for wikihow_instance in all_data:
        if diff_nouns:
            print("work with diff-noun-modifications.")
            source_tokenized = [pair[0]
                                for pair in wikihow_instance['Source_tagged']]
            target_tokenized = [pair[0]
                                for pair in wikihow_instance['Target_Tagged']]
        else:
            print("work with same-noun modifications.")
            source_tokenized = [pair[0]
                                for pair in wikihow_instance['Source_Line_Tagged']]
            target_tokenized = [pair[0]
                                for pair in wikihow_instance['Target_Line_Tagged']]
        if source:
            print("append the source .... ")
            documents.append(source_tokenized)
        else:
            print("append the target ... ")
            documents.append(target_tokenized)
    return documents


def main():
    # get all source instances for diff and same noun modifications
    path_to_train_diff, path_to_dev_diff, path_to_test_diff = get_paths(
        different_nouns=True)
    path_to_train_same, path_to_dev_same, path_to_test_same = get_paths(
        different_nouns=False)
    diff_noun_source = get_data(
        path_to_train_diff, path_to_dev_diff, path_to_test_diff, diff_nouns=True, source=False)
    same_noun_source = get_data(
        path_to_train_same, path_to_dev_same, path_to_test_same, diff_nouns=False, source=False)

    # merge data
    all_sentences = diff_noun_source + same_noun_source

    total_score = 0
    bar = Bar('Processing ', max=len(all_sentences))
    for sent in all_sentences:
        bar.next()
        score_dict = check_discourse_maches(sent)

        total_score += score_dict["score"]
        print(score_dict)
        print('----------')
    bar.finish()

    print("Total score for source: ")
    print(total_score)


main()
