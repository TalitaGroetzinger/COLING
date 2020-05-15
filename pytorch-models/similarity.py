import torchtext.vocab
from nltk.tokenize import word_tokenize
import torch
import nltk
from nltk.corpus import stopwords
list_of_stopwords = set(stopwords.words('english'))

glove = torchtext.vocab.GloVe(name='6B', dim=100)


def compute_sim(word1, word2):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    word1 = word1.unsqueeze(1)
    word2 = word2.unsqueeze(1)
    output = cos(word1, word2)
    return output


def get_vector(embeddings, word):
    if word in embeddings.stoi and word.lower() not in list_of_stopwords:
        return embeddings.vectors[embeddings.stoi[word]]
    else:
        return torch.ones(100)


def compute_sentence_vec(sent):
    """
      Input: sent in string format
      Output: a stacked vector 

    """
    sentence_emb = []

    for word in word_tokenize(sent):
        embedding = get_vector(glove, word.lower())
        sentence_emb.append(embedding)

    stacked_tensor = torch.stack(sentence_emb)

    # apply product to do the product
    return torch.prod(stacked_tensor, 0)


def compute_sentence_similarity(sentence1, sentence2, return_tensor=False):
    vec1 = compute_sentence_vec(sentence1)
    vec2 = compute_sentence_vec(sentence2)
    if return_tensor:
        return compute_sim(vec1, vec2)
    else:
        tensor_cos_sim = compute_sim(vec1, vec2)
        return tensor_cos_sim.item()


def main():
    example_sent1 = "This is a sentence."
    example_sent2 = "This is a."

    res = compute_sentence_similarity(example_sent1, example_sent2)
    print(res)


if __name__ == '__main__':
    main()
