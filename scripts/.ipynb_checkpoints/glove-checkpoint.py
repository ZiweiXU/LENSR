import pickle as pk

import bcolz
import numpy as np


def load_glove_to_binary(path: str, dimension: int = 50, identifier='6B'):
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(
        np.zeros(1),
        rootdir='{}/glove.{}.{}d.dat'.format(path, identifier, dimension),
        mode='w')
    with open('{}/glove.{}.{}d.txt'.format(path, identifier, dimension), 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    vectors = bcolz.carray(
        vectors[1:].reshape((400000, dimension)),
        rootdir='{}/glove.{}.{}d.dat'.format(path, identifier, dimension),
        mode='w')
    vectors.flush()
    pk.dump(words, open('{}/glove.{}.{}d.words.pkl'.format(path, identifier, dimension), 'wb'))
    pk.dump(word2idx, open('{}/glove.{}.{}d.idx.pkl'.format(path, identifier, dimension), 'wb'))


def load_glove_as_dict(path: str, dimension: int = 50, identifier='42B'):
    vectors = bcolz.open('{}/glove.{}.{}d.dat'.format(path, identifier, dimension))[:]
    words = pk.load(open('{}/glove.{}.{}d.words.pkl'.format(path, identifier, dimension), 'rb'))
    word2idx = pk.load(open('{}/glove.{}.{}d.idx.pkl'.format(path, identifier, dimension), 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove


if __name__ == "__main__":
    load_glove_to_binary('./', dimension=50, identifier='6B')
