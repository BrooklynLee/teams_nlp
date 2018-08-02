from gensim.models import FastText, KeyedVectors
import random
import re
import nltk


def similar(self):
    global model
    model = KeyedVectors.load_word2vec_format('./data/wiki-news/wiki-brooklyn.bin', binary=True, unicode_errors='ignore')
#    sim2 = []
#    sim1 = model.most_similar(self, topn = 25)
#    sim3 = [sim2.append(word) for word, score in sim1]

############## KEYWORD EXTRACTION ##################
    token = nltk.word_tokenize(self)
    pos = nltk.pos_tag(token)

    keyword_list = []
    mostRelated = []

    for s in pos:
        if s[1] in ['NNP', 'JJ', 'NN']:
            keyword_list.append(s[0])

############## SIMILAR WORDS ##################


    for i in keyword_list:
        if i in model.vocab:
            mostRelated_with_score = model.most_similar(positive=i, topn=10)
            for word, score in mostRelated_with_score:
                word2 = re.sub('[-=.#/?:$}]', '', word)
                mostRelated.append(word2)

    return mostRelated


############## MARKOV_CHAIN ##################


# def markov(self):
