import pickle
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from tqdm import tqdm

# use this script to do everything but for intermediate insertions
# work with the pandas dataframe


def tag_data(x):
    x = word_tokenize(x)
    return nltk.pos_tag(x)

# todo: function to check if the length is the same for source-target + POS tags are the same


def main():
    path_to_df = "/Users/talitaanthonio/Documents/PhD/year1/wiki-how-scripts/data/wikihow_instructional_text_ordered_no_cycle_v6.pickle"
    df = pd.read_pickle(path_to_df)
    df = df.head()
    tqdm.pandas()
    df['Source_Line_Tagged'] = df['Source_Line'].progress_apply(tag_data)
    df['Target_Line_Tagged'] = df['Target_Line'].progress_apply(tag_data)
    df.to_pickle('everything_tagged.pickle')
    # apply POS tags

    # flatten the dictionary


main()
