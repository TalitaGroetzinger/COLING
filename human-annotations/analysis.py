import pandas as pd


def get_answers(revised_first, base_first, df):

    # count how often the annotator answered that sentence A is better (because A = revised, B = base)
    revised_first_d = dict(
        revised_first["Answer.answer-a"].value_counts())

    # count how often the annotator answered that Sentence B is better (because B = revised, A = base)
    base_first_d = dict(
        base_first["Answer.answer-b"].value_counts())

    correct_for_base = base_first_d[True]
    correct_for_revised = revised_first_d[True]

    print("correct for base first {0}".format(correct_for_base))
    print("correct for revised first {0}".format(correct_for_revised))

    total_correct = correct_for_base + correct_for_revised
    length_df = len(base_first) + len(revised_first)

    print("total correct {0} {1}".format(
        total_correct, total_correct/length_df))

    # check how many unkowns

    annotator_not_sure = df.loc[df['Answer.not-sure'] == True]
    print("not sure {0}".format(len(annotator_not_sure)/len(df)))


def main():
    df = pd.read_csv('alok_results.tsv', sep='\t')
    df = df.drop(columns=[
        'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
        'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
        'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
        'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Input.Title', 'Approve', 'Reject'], axis=1)

    revised_first = df.loc[df['Input.Info'] == 'revised']
    base_first = df.loc[df['Input.Info'] == 'base']

    print("Length for revised: {0}".format(len(revised_first)))
    print("Length for base: {0}".format(len(base_first)))
    print("total: {0}".format(len(df)))
    print("\n")

    """
    perc_true_revised, perc_false_revised = get_individual_answers(
        revised_first)
    perc_true_base, perc_false_base = get_individual_answers(
        base_first, revised=False)

    total_correct = perc_true_revised + perc_true_base
    total_incorrect = perc_false_revised + perc_false_base

    print(total_correct)
    print(total_incorrect)
    """
    get_answers(revised_first, base_first, df)


main()
