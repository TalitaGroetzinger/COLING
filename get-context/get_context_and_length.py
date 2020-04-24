import json

with open('../classification-scripts/subset-train.json', 'r') as json_in:
    content = json.load(json_in)


def get_full_article(filename, collection, line_nr='400'):
    current_line = collection[filename][line_nr]
    sentence_line_nrs_in_article = [
        key for key, _ in collection[filename].items()]
    list_positions = [i for i in range(len(sentence_line_nrs_in_article))]

    line_nr_pos = sentence_line_nrs_in_article.index(str(line_nr))
    # get everything before timestamp plus the content on line_nr

    left_article_until_timestamp = []
    for i in range(line_nr_pos-0):
        key = i+1
        index_in_dict = sentence_line_nrs_in_article[key]

        if '## Timestamp' not in content[filename][index_in_dict]:
            line = "{0} {1}".format(
                index_in_dict, content[filename][index_in_dict])
            left_article_until_timestamp.append(line)
        else:
            print(left_article_until_timestamp)


def main():
    subset_filenames = ['Be_Resourceful.txt', 'Create_a_Well_Rounded_Approach_for_Getting_Rid_of_Acne.txt', 'Get_a_Big_Warhammer_Bitz_Box.txt', 'Improve_Your_Class_Ranking.txt', 'Clean_a_Sticking_Delta_Soap_Dispenser.txt',
                        'Graciously_Accept_an_Unattractive_Gift_from_the_in_Laws.txt', 'Pay_Off_Student_Loans.txt', "Edit_a_Friend's_Essay.txt", 'Write_Apocalyptic_Stories.txt', 'Make_Your_Room_Look_Nice.txt', 'Deal_with_Sexual_Abuse.txt', 'Have_a_Larger_Vocabulary.txt']
    for elem in subset_filenames:
        get_full_article(elem, content)
        break


main()
