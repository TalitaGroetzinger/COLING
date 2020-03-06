import json
from parse_db import read_ppdb
from progress.bar import Bar


def read_data(path_to_json_file):
    with open(path_to_json_file, 'r') as f:
        content = json.load(f)
    return content


def main():
    content = read_data('./data/noun_corrections.json')
    print(len(content))
    print("read pdb")
    pdb = read_ppdb('./data/ppdb/ppdb-xxxl-lexical.txt')
    bar = Bar('Processing', max=len(content))
    total = 0
    matches = 0
    match_per_sent_total = 0
    for elem in content:
        matches_in_pair = 0
        for pair in elem['Differences']:
            total += 1
            source = pair[0]
            target = pair[1]
            key_to_look_for = "{0}#{1}".format(
                source[0].lower(), target[0].lower())
            try:
                print(pdb[key_to_look_for])
                print('\n')
                matches += 1
                matches_in_pair += 1
            except KeyError:
                continue

        if matches_in_pair > 0:
            match_per_sent_total += 1
        bar.next()
    bar.finish()
    total_matches = matches / total
    print("Total Pairs in Corpus: ", len(content))
    print("Sents with matches:", match_per_sent_total)
    print('Percentage:', (match_per_sent_total/len(content)))
    print("-------------------------------------")
    print("Total differences: ", total)
    print("Matches: ", matches)
    print("Perc: ", total_matches)


main()
