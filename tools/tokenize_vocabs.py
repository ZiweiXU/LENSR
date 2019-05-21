import sys

sys.path.append('.')
sys.path.append('../')

import pickle as pk
import json
import pdb

from model.Misc.Glove import load_glove_as_dict
from model.Misc.Conversion import POS_REL_NAMES, POS_REL_NAMES_FULL

glove = load_glove_as_dict('../dataset/glove', 50, identifier='6B')
objs = pk.load(open('../dataset/VRD/objects.pk','rb'))
pres = pk.load(open('../dataset/VRD/predicates.pk','rb'))
for i in POS_REL_NAMES:
    pres.append(i)
pres.append('_exists')
pres.append('_unique')

tokenizer = {'token2vocab': {}, 'vocab2token': {}}
current_token = 0
word_vector = []


def add_token(tokenizer, object, current_token, word_vector, glove):
    tokenizer['vocab2token'][object] = current_token
    tokenizer['token2vocab'][current_token] = object
    current_token += 1
    if object == '"walk"':
        object = 'walk'
    try:
        word_vector.append(glove[object.lower()])
    except Exception as e:
        print(object)
        word_vector.append(glove['<unk>'])
    return tokenizer, current_token


for obj in objs:
    if len(obj.split()) == 1:
        tokenizer, current_token = add_token(tokenizer, obj.split()[0], current_token, word_vector,
                                             glove)
    elif len(obj.split()) > 1:
        for w in obj.split():
            if w not in tokenizer['vocab2token'].keys():
                tokenizer, current_token = add_token(tokenizer, w, current_token, word_vector,
                                                     glove)

for pre in pres:
    if pre in POS_REL_NAMES:
        pre_1 = POS_REL_NAMES_FULL[pre]
    else:
        pre_1 = pre

    if len(pre_1.split()) == 1:
        if pre_1.split()[0] not in ['_exists', '_unique']:
            tokenizer, current_token = add_token(tokenizer, pre_1.split()[0], current_token,
                                                 word_vector, glove)
        else:
            tokenizer, current_token = add_token(tokenizer, pre_1.split()[0][1:], current_token,
                                                 word_vector, glove)
    elif len(pre_1.split()) > 1:
        for w in pre_1.split():
            if w not in tokenizer['vocab2token'].keys():
                tokenizer, current_token = add_token(tokenizer, w, current_token, word_vector,
                                                     glove)

pk.dump(tokenizer, open('../dataset/VRD/tokenizers.pk', 'wb'))
pk.dump(word_vector, open('../dataset/VRD/word_vectors.pk', 'wb'))
