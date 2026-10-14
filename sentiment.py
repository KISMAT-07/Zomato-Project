"""Shared review preprocessing for training and serving."""
import re
import string

# Keep negations: they carry sentiment (e.g. 'not good').
STOPWORDS = frozenset('a an the and or but if while with to from for of in on at by as is am are was were be been being this that these those it its i me my we our you your he she they them his her their there here so very too also can could would should will just about into than then over under again once do does did only own same such have has had having after before more most some any all few each other what which who whom when where why how'.split())

def normalize_review(text):
    text = str(text).lower().replace('’', "'")
    for short, full in [("can't", 'can not'), ("cannot", 'can not'), ("won't", 'will not'), ("n't", ' not'), ("'re", ' are'), ("'s", ' is'), ("'d", ' would'), ("'ll", ' will'), ("'ve", ' have'), ("'m", ' am')]:
        text = text.replace(short, full)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    text = text.translate(str.maketrans({c: ' ' for c in string.punctuation}))
    return ' '.join(word for word in text.split() if word not in STOPWORDS and len(word) > 1)
