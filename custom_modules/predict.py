from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from embeddinger import embed_bert_cls


def classify(text, regress, tokenizer, model):
    x = embed_bert_cls(text, model, tokenizer) 
    return regress.predict(x.reshape(1,-1))

def classify_texts(texts, regress, tokenizer, model):
    sents = []
    for text in tqdm(texts):
        res = classify(text, regress = regress, tokenizer = tokenizer, model = model)[0]
        sents.append(1 - res)

    return sents

