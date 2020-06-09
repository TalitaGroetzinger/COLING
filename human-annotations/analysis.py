# Input: df with columns ['Input.Title', 'Input.Batch_ID', 'Input.Line1', 'Input.Line2',
# 'Input.Context1', 'Input.Context2', 'Input.Info', 'Input.Differences',
# 'Answer.answer-a', 'Answer.not-sure', 'Answer.answer-b',
# 'Answer.annotation-comments']
# Step 1: add another column with the batch_id of the source
# Step 2: add a column that mentions whether the annotator annotated the revised version as the better one "CORRECT"

import pandas as pd
import pickle

with open('entailment_relations_development_set.pickle', 'rb') as pickle_in:
    entailment_relations = pickle.load(pickle_in)


def get_answers(revised_is_A, base_is_A, df):

    # count how often the annotator answered that sentence A is better (because A = revised, B = base)
    revised_is_A_dict = dict(
        revised_is_A["Answer.answer-a"].value_counts())
    print("Results for Sentence A = revised")
    print(revised_is_A_dict)

    # count how often the annotator answered that Sentence B is better (because B = revised, A = base)
    base_is_A_dict = dict(
        base_is_A["Answer.answer-b"].value_counts())

    print("results for answer b = revised")
    print(base_is_A_dict)

    # count not sure
    not_sure = df.loc[df['Answer.not-sure'] == True]
    print("Instances where annotator was not sure: {0} ".format(len(not_sure)))


def make_new_df(df, revised_first=True):
    new_df = {"Base_batch_id": [], "Revised_Marked_as_better": []}
    cases_correct = []
    num = 0
    if revised_first:
        for index, row in df.iterrows():
            if row['Answer.answer-a'] == True and row['Input.Info'] == 'revised':
                cases_correct.append("CORRECT")
            else:
                if row['Answer.not-sure'] == True:
                    cases_correct.append("NOT SURE")
                else:
                    cases_correct.append("INCORRECT")
        return cases_correct
    else:
        for index, row in df.iterrows():
            if row['Answer.answer-b'] == True and row['Input.Info'] == 'base':
                cases_correct.append("CORRECT")
                num += 1
            else:
                if row['Answer.not-sure'] == True:
                    cases_correct.append("NOT SURE")
                else:
                    cases_correct.append("INCORRECT")

        return cases_correct


def get_batch_id(df):
    source_batch_id = []
    for index, row in df.iterrows():
        source_text, source_batch, target_text, target_batch = row['Input.Batch_ID'].split(
        )
        source_batch_id.append(source_batch)
    return source_batch_id


def get_list_of_relations(df):
    list_of_entailment_relations = []
    for index, row in df.iterrows():
        source_id = row["Source_batch_id"]

        entailment_rel = entailment_relations[int(source_id)]
        list_of_entailment_relations.append(entailment_rel)
    return list_of_entailment_relations


def main():
    df = pd.read_csv('./alok_results/alok_results_new.tsv', sep='\t')
    # df['Answer.answer-b'] = df['Answer.answer-b'].replace(['TRUE ', 'TRUE'])
    print(df.columns)

    # voeg batch id van de source toe
    source_batch_id = get_batch_id(df)

    df['Source_batch_id'] = source_batch_id

    entailment_rels = get_list_of_relations(df)
    df['Entailment_rel'] = entailment_rels

    # get the case where sentence A = revised
    revised_is_a = df.loc[df['Input.Info'] == 'revised']
    # get the cases where sentence A = base
    base_is_a = df.loc[df['Input.Info'] == 'base']

    print("Length for revised first : {0}".format(len(revised_is_a)))
    print("Length for base first: {0}".format(len(base_is_a)))
    print("total: {0}".format(len(df)))
    print("\n")

    get_answers(revised_is_a, base_is_a, df)

    cases_correct_revised_first_row = make_new_df(
        revised_is_a, revised_first=True)

    revised_is_a["Annotator1_Answer"] = cases_correct_revised_first_row

    cases_correct_base_first_row = make_new_df(base_is_a, revised_first=False)

    base_is_a["Annotator1_Answer"] = cases_correct_base_first_row

    final_df = pd.concat([revised_is_a, base_is_a], ignore_index=True)
    print(final_df)
    final_df.to_csv('./alok_results/alok_results_transformed.tsv',
                    sep='\t', index=False)

    revised_annotated_as_better = final_df.loc[final_df['Annotator1_Answer'] == 'CORRECT']
    # get the cases where sentence A = base
    not_sure = final_df.loc[final_df['Annotator1_Answer'] == 'NOT SURE']
    base_annotated_as_better = final_df.loc[final_df['Annotator1_Answer'] == 'INCORRECT']

    revised_annotated_as_better.to_csv(
        './alok_results/alok_revised_is_better.tsv', sep='\t', index=False)
    not_sure.to_csv('./alok_results/alok_not_sure.tsv', sep='\t', index=False)
    base_annotated_as_better.to_csv(
        './alok_results/alok_base_is_better.tsv', sep='\t', index=False)


main()
