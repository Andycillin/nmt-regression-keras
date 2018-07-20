'''
Author: Thien Phuc Tran, Hong Hai Le
'''
import gensim
import numpy as np
from utils import load_dataset, pad_sentence
from keras.models import load_model
from model import softmax_cosine_distance_loss, softmax_mse_distance_loss

# embedding
words_limit = 500000
sentence_length_limit = 20
model = load_model('./seq2seq_regressor_trained.h5',
                   custom_objects={'softmax_mse_distance_loss': softmax_mse_distance_loss})

print("Importing Word2Vec model, limit %d words" % words_limit)
en_model = gensim.models.KeyedVectors.load_word2vec_format('crawl-300d-2M.vec', limit=words_limit)
print("Word2Vec English imported")
de_model = gensim.models.KeyedVectors.load_word2vec_format('wiki.de.vec', limit=words_limit)
print("Word2Vec German imported")

undefined_word_vec = np.ones((300,), dtype=np.float32)
endtoken_vec = np.zeros((300,), dtype=np.float32)

while (True):
    x = input("Enter english:")
    de_line_vecs = [en_model[w] if w in en_model.vocab else undefined_word_vec for w in x.split()]
    de_line_vecs = pad_sentence(de_line_vecs, sentence_length_limit, endtoken_vec)

    predictions = model.predict(np.array([de_line_vecs]))
    reg_pred = predictions[:, :, :300][0]
    softmax_pred = predictions[:, :, -1][0]
    outputlist = [de_model.most_similar([reg_pred[i]])[0][0] if softmax_pred[i] >= 0.5 else ' ' for i in
                  range(20)]
    outputlist2 = [de_model.most_similar([reg_pred[i]])[0][0] for i in
                   range(20)]

    output = ' '.join(outputlist)
    output2 = ' '.join(outputlist2)
    print('w/ Classif: %s' % output)
    print('w/o Classif: %s' % output2)
