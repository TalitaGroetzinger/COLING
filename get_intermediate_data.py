import pickle
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm
from wikihowtools.add_linguistic_info import compute_char_distance

# use this script to do everything but for intermediate insertions
# work with the pandas dataframe


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
    collection = []
    for index, row in df.iterrows():
        differences = []
        source_tagged = row['Source_Line_Tagged']
        target_tagged = row['Target_Line_Tagged']
        if len(source_tagged) == len(target_tagged):
            for source, target in zip(source_tagged, target_tagged):
                if source[0] != target[0]:
                    distance = compute_char_distance(
                        source[0], target[0])
                    if distance > edit_distance_value:
                        differences.append([source, target])
        if differences != []:
            d = {"Filename": row['Article_Name'], "Source_Line": row['Source_Line'], "Target_Line": row["Target_Line"], "Differences": differences,
                 "Source_Line_Tagged": source_tagged, "Target_Line_Tagged": target_tagged

                 }
            collection.append(d)
    return collection


def main():
    path_to_df = "/Users/talitaanthonio/Documents/PhD/year1/wiki-how-scripts/data/wikihow_instructional_text_ordered_no_cycle_v6.pickle"
    df = pd.read_pickle(path_to_df)
    tqdm.pandas()
    df['Source_Line_Tagged'] = df['Source_Line'].progress_apply(tag_data)
    df['Target_Line_Tagged'] = df['Target_Line'].progress_apply(tag_data)
    df.to_pickle(
        'wikihow_instructional_text_ordered_no_cycle_v6_tagged_source_target.pickle')


main()
