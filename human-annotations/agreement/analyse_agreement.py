import pandas as pd

df = pd.read_csv('annotator_comparison.tsv', sep='\t')
print(df)

# get the cases where both have provided revised=better
correct_annotator1 = df['Annotator1_Answer'] == 'CORRECT'
correct_annotator2 = df['Annotator2_Answer'] == 'CORRECT'
# 153 x 12
both_revised_is_better = df[correct_annotator1 & correct_annotator2]
both_revised_is_better.to_csv(
    'both_revised_is_better.tsv', index=False, sep='\t')

# get the cases where the annotators are both not sure
not_sure_annotator1 = df['Annotator1_Answer'] == 'NOT SURE'
not_sure_annotator2 = df['Annotator2_Answer'] == 'NOT SURE'
# 142 x 12
both_not_sure = df[not_sure_annotator1 & not_sure_annotator2]
both_not_sure.to_csv('both_not_sure.tsv', index=False, sep='\t')

# get the cases where both annotators said that the revised is better
incorrect_annotator1 = df['Annotator1_Answer'] == 'INCORRECT'
incorrect_annotator2 = df['Annotator2_Answer'] == 'INCORRECT'
# 19 x 12
both_incorrect = df[incorrect_annotator1 & incorrect_annotator2]
both_incorrect.to_csv('both_base_is_better.tsv', index=False, sep='\t')
