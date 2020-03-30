import json
import pickle

markers = '../data/discourse_markers.pickle'
with open(markers, 'rb') as pickle_in:
    markers = pickle.load(pickle_in)


with open('../data/subset.json', 'r') as json_in:
    subset = json.load(json_in)


def check_discourse_maches(wikihow_instance):
    total = 0
    unigram_matches = 0
    bigram_matches = 0
    trigram_matches = 0
    fourgram_matches = 0
    fivegram_matches = 0
    print("-----")
    # Later zal dit meteen het document zijn/de tokens.
    tokens = [pair[0].lower() for pair in wikihow_instance['Source_Tagged']]
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


for elem in subset:
    score = check_discourse_maches(elem)
    print(score)
    print('----------')
