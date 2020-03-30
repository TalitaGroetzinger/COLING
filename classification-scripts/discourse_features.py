import pickle


def clean_list(dict_with_lists):
    d = {}
    for key, _ in dict_with_lists.items():
        if dict_with_lists[key] != []:
            d[key] = dict_with_lists[key]
    return d


def make_first_dict(discourse_markers_list):
    discourse_markers = []
    for line in discourse_markers_list:
        line = line.strip('\n')
        line = line.strip()
        discourse_markers.append(line.lower())
    discourse_markers_uniq_list = list(set(discourse_markers))
    discourse_markers_uniq = [elem.split()
                              for elem in discourse_markers_uniq_list]

    d = {}
    for marker in discourse_markers_uniq:
        d[marker[0]] = {"unigrams": [], "bigrams": [],
                        "trigrams": [], "fourgrams": [],  "fivegrams": []}
    return d, discourse_markers_uniq


def add_grams(d, discourse_markers_list):
    for words in discourse_markers_list:
        key_in_dict = words[0]
        if len(words) == 1:
            d[key_in_dict]['unigrams'].append(words)
        elif len(words) == 2:
            d[key_in_dict]['bigrams'].append(words)
        elif len(words) == 3:
            d[key_in_dict]['trigrams'].append(words)
        elif len(words) == 4:
            d[key_in_dict]['trigrams'].append(words)
        else:
            d[key_in_dict]['fourgrams'].append(words)
    return d


def main():
    path_to_discourse_markers = '../data/discourse_markers_all.txt'
    with open(path_to_discourse_markers, 'r') as file_in:
        discourse_markers_list = file_in.readlines()

    d, discourse_markers_list_uniq = make_first_dict(discourse_markers_list)

    new = add_grams(d, discourse_markers_list_uniq)

    with open('../data/discourse_markers.pickle', 'wb') as pickle_in:
        pickle.dump(new, pickle_in)


main()
