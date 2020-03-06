import pickle
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm
from wikihowtools.add_linguistic_info import compute_char_distance
import json
from progress.bar import Bar


def tag_data(x):
    x = word_tokenize(x)
    return nltk.pos_tag(x)


def filter_df(df, edit_distance_value=2):
    """
      Input: a dataframe containing Source_Line_Tagged and Target_Line_Tagged.
      Output: a list with dictionaries containing only the base-revision where:
      - the POS-tags in the base and revised sentence are the same
      - a word (or more words) that has changed into another word and this change has a minimum character-based edit-distance of 3

    """
    bar = Bar('Processing', max=len(df))
    collection = []
    for index, row in df.iterrows():
        bar.next()
        differences = []
        tags = []
        tags_source = [word_pos_pair[1]
                       for word_pos_pair in row['Source_Line_Tagged']]
        tags_target = [word_pos_pair[1]
                       for word_pos_pair in row['Target_Line_Tagged']]
        if tags_source == tags_target:
            for source, target in zip(row['Source_Line_Tagged'], row['Target_Line_Tagged']):
                if source[0] != target[0]:
                    distance = compute_char_distance(
                        source[0], target[0])
                    if distance > edit_distance_value:
                        if 'NN' == source[1] or 'NNS' == source[1] or 'NNP' == source[1]:
                            differences.append([source, target])
                            assert source[1] == target[1]
                            tags.append(source[1])
        if differences != []:
            d = {"Filename": row['Article_Name'], "Source_Line_Tagged": row['Source_Line_Tagged'],
                 "Target_Line_Tagged": row['Target_Line_Tagged'], "Differences": differences,  "Tags": tags, "Revision_Length": row['Revision_Length']}
            collection.append(d)
    bar.finish()
    return collection


def main():
    df = pd.read_pickle(
        './data/wikihow_instructional_text_ordered_no_cycle_v6_tagged_source_FINAL.pickle')
    # df = pd.read_pickle('subset.pickle')
    print("Make corrections ...")
    corrections = filter_df(df)
    print(len(corrections))

    print("Save file ... ")
    with open('all_corrections_wikihow_v6.json', 'w') as f:
        json.dump(corrections, f)


main()
