import json
import pandas as pd


def get_list_of_files():
    with open('../classification-scripts/classification-data/DIFF-NOUN-MODIFICATIONS.json', 'r') as json_in:
        diff_noun_modifications = json.load(json_in)

    with open('diff_noun_modifications.tsv', 'w') as tsv_file:
        header = "{0}\t{1}\t{2}".format(
            'Filename', 'Source_Tagged', 'Target_Tagged')
        tsv_file.write(header)
        for elem in diff_noun_modifications:
            line_to_write = "{0}\t{1}\t{2}\n".format(
                elem['Filename'], elem['Source_tagged'], elem['Target_Tagged'])
            tsv_file.write(line_to_write)


def main():
    # make a dataframe of the complete wikihow dataset
    path_to_full = '../../../corpora/wikihow_instructional_text_ordered_no_cycle_v6.txt'
    df = pd.read_csv(path_to_full, delimiter="\t", quoting=3)

    # get only the rows with a specific filename
    with open('../data/diff_noun_modification_files.txt', 'r') as file_in:
        list_of_filenames = [line.strip('\n') for line in file_in.readlines()]
        list_of_filenames_unique = set(list_of_filenames)

    filtered_df = df.loc[df['Article_Name'].isin(list_of_filenames_unique)]

    article_names_in_df = set(filtered_df['Article_Name'].tolist())
    # Minus 1 here because the first line of diff_noun_modification_files.txt is the header.
    try:
        assert len(article_names_in_df) == len(list_of_filenames_unique)-1
    except AssertionError:
        print(len(article_names_in_df))
        print(len(list_of_filenames_unique))

    filtered_df.to_pickle('potential-diff-noun-modifications.pickle')


main()
