import json
import re
from nltk.tokenize import word_tokenize


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def compute_length(tokenized_text):
    num_of_tokens = len(tokenized_text)
    unique_tokens = list(set(tokenized_text))
    num_of_unique_tokens = len(unique_tokens)
    return {"Type-token-ratio": num_of_unique_tokens/num_of_tokens, "Length": num_of_tokens}


def get_left(article_database, filename, line_nr, sentence_nrs):
    above_line_nrs = list(filter(lambda sentence_nr: int(
        sentence_nr) <= int(line_nr), sentence_nrs))[::-1]

    before_lines = []
    for line_nr in above_line_nrs:
        line = content[filename][line_nr]
        # check if there is a timestamp
        if '## Timestamp' in line:
            print("Timestamp found ..", line_nr)
            break
        before_lines.append(line)
    return before_lines[::-1]


def get_right(article_database, filename, line_nr, sentence_nrs):
    after_line_nrs = filter(lambda sentence_nr: int(
        sentence_nr) > int(line_nr), sentence_nrs)

    after_lines = []

    for line_nr in after_line_nrs:
        line = content[filename][line_nr]
        if '## Timestamp' in line:
            print("Timestamp found ..", line_nr)
            break
        after_lines.append(line)
    return after_lines


def get_full_article(article_database, filename, line_nr='400'):
    sentence_nrs = [key for key in article_database[filename].keys()]
    # Take everything above the content that you want including the source or target line.
    left_content = get_left(article_database, filename, line_nr, sentence_nrs)

    # do the same procedure for the right
    right_content = get_right(
        article_database, filename, line_nr, sentence_nrs)

    full_article = left_content + right_content
    # clean the html tags
    full_article_cleaned = [remove_html_tags(sents) for sents in full_article]
    tokenized = []
    for sents in full_article_cleaned:
        tokens = word_tokenize(sents)
        tokenized += tokens
    res = compute_length(tokenized)
    return res


if __name__ == "__main__":
    with open('../classification-scripts/subset-train.json', 'r') as json_in:
    content = json.load(json_in)

    subset_filenames = ['Be_Resourceful.txt', 'Create_a_Well_Rounded_Approach_for_Getting_Rid_of_Acne.txt', 'Get_a_Big_Warhammer_Bitz_Box.txt', 'Improve_Your_Class_Ranking.txt', 'Clean_a_Sticking_Delta_Soap_Dispenser.txt',
                        'Graciously_Accept_an_Unattractive_Gift_from_the_in_Laws.txt', 'Pay_Off_Student_Loans.txt', "Edit_a_Friend's_Essay.txt", 'Write_Apocalyptic_Stories.txt', 'Make_Your_Room_Look_Nice.txt', 'Deal_with_Sexual_Abuse.txt', 'Have_a_Larger_Vocabulary.txt']
    for filename in subset_filenames:
        get_full_article(content, filename)
        break
