from konlpy.tag import Kkma, Okt
import gensim
from gensim.models import Word2Vec
import random
import re
import numpy as np

okt = Okt()
kkma = Kkma()
model = None


def similar(self):

    global model

    if model is None:
        model = Word2Vec.load('./data/wiki-news/ko.bin')



############## KEYWORD EXTRACTION ##################
    keyword_list = []
    mostRelated = []

    pos = okt.pos(self)
    for s in pos:
        if s[1] in ['Noun', 'Verb']:
            keyword_list.append(s[0])
############## SIMILAR WORDS ##################


    for i in keyword_list:
        if i in model.wv.vocab:
            mostRelated_with_score = model.most_similar(positive=i, topn=10)
            for word, score in mostRelated_with_score:
                word2 = re.sub('[-=.#/?:$}]', '', word)
                mostRelated.append(word2)



    return mostRelated

