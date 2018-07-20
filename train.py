'''
Author: Thien Phuc Tran
'''
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from model import build_model
import callbacks
from math import ceil

file_path = "ger_eng_dataset.pkl"
bytes_in = bytearray(0)
n_bytes = 2 ** 31
max_bytes = 2 ** 31 - 1
input_size = os.path.getsize(file_path)
print("Loading dataset...")
with open(file_path, 'rb') as f_in:
    for _ in range(0, input_size, max_bytes):
        bytes_in += f_in.read(max_bytes)
datasetX, datasetY = pickle.loads(bytes_in)
print("Dataset loaded")
print("Num of samples: %s" % (len(datasetX)))

# Split dataset
datasetX = np.array(datasetX[:20000], dtype=np.float64)
datasetY = np.array(datasetY[:20000], dtype=np.float64)

# Calc masks
_sum = np.sum(datasetY, axis=-1)  # (n, 20, 1)
mask = (_sum[:] != 0).astype(int)
mask = np.expand_dims(mask, axis=2)
datasetY = np.concatenate((datasetY, mask), axis=-1)

x_train, x_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2, random_state=1)
print("Training samples: %s" % len(x_train))
print("Validation samples: %s" % len(x_test))

# embedding
words_limit = 500000
sentence_length_limit = 20
batch_size = 32
steps_per_epoch = ceil(x_train.shape[0] / batch_size)

model = build_model(sentence_length_limit, 300, encoder_units=50)
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=True)

save_dir = os.path.join(os.getcwd(), 'saved_models')

for i in range(1, 101):
    model.fit(x_train, y_train,
              epochs=100,
              batch_size=batch_size,
              callbacks=[callbacks.VisualizationCallback(steps_per_epoch, 'seq2seq Regressor MSE')],
              validation_data=(x_test, y_test))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name = 'seq2seq_regressor_mse_trained2_' + str(50 * i) + '.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
