'''
Author: Thien Phuc Tran, Hong Hai Le
'''
import gensim
import numpy as np
import pickle
from utils import load_dataset, pad_sentence

# Load data
en_lines, de_lines = load_dataset('deu.txt')

# embedding
words_limit = 500000
sentence_length_limit = 20

print("Importing Word2Vec model, limit %d words" % words_limit)
en_model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', limit=words_limit)
print("Word2Vec English imported")
de_model = gensim.models.KeyedVectors.load_word2vec_format('wiki.de.vec', limit=words_limit)
print("Word2Vec German imported")

undefined_word_vec = np.ones((300,), dtype=np.float32)
endtoken_vec = np.zeros((300,), dtype=np.float32)

en_vecs = []
de_vecs = []
for i in range(0, len(en_lines)):
    en_line = en_lines[i]
    de_line = de_lines[i]
    if len(en_line.split()) <= sentence_length_limit and len(de_line.split()) <= sentence_length_limit:
        en_line_vecs = [en_model[w] if w in en_model.vocab else undefined_word_vec for w in en_line.split()]
        en_line_vecs = pad_sentence(en_line_vecs, sentence_length_limit, endtoken_vec)
        en_vecs.append(en_line_vecs)

        de_line_vecs = [de_model[w] if w in de_model.vocab else undefined_word_vec for w in de_line.split()]
        de_line_vecs = pad_sentence(de_line_vecs, sentence_length_limit, endtoken_vec)
        de_vecs.append(de_line_vecs)

file_path = "ger_eng_dataset.pkl"
n_bytes = 2 ** 31
max_bytes = 2 ** 31 - 1
## write
bytes_out = pickle.dumps([en_vecs, de_vecs])
with open(file_path, 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx + max_bytes])
print("exported")
