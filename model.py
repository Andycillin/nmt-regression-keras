'''
Author: Thien Phuc Tran
'''
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers import LSTM, RepeatVector, Dense, Activation, Input, Concatenate, GRU
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization


def build_model(sentence_length_limit=20, word2vec_dim=300, encoder_units=50):
    decoder_units = encoder_units * 2
    RNNLayer = GRU
    input_shape = (sentence_length_limit, word2vec_dim)
    optimizer = Adam(lr=0.001, decay=0.0005)

    # model = Sequential()
    # Encoder:
    # model.add(Bidirectional(RNNLayer(encoder_units), input_shape=input_shape))
    inp_shape = Input(shape=input_shape)
    x = Bidirectional(RNNLayer(encoder_units), input_shape=input_shape)(inp_shape)
    # model.add(BatchNormalization())
    x = BatchNormalization()(x)
    # The RepeatVector-layer repeats the input n times
    # model.add(RepeatVector(sentence_length_limit))
    x = RepeatVector(sentence_length_limit)(x)
    # Decoder:
    # model.add(Bidirectional(LSTM(decoder_units, return_sequences=True)))
    x = Bidirectional(RNNLayer(decoder_units, return_sequences=True))(x)
    # model.add(BatchNormalization())
    x = BatchNormalization()(x)
    # to dense layer
    # model.add(TimeDistributed(Dense(word2vec_dim)))
    reg_out = TimeDistributed(Dense(word2vec_dim), name='reg_layer')(x)  # output (n,20,300)
    softmax_out = TimeDistributed(Dense(1, activation='sigmoid'), name='softmax_layer')(x)  # output(n,20,2)
    # model.compile(loss=cosine_distance_loss, optimizer=optimizer, metrics=['accuracy'])
    pred = Concatenate(axis=2)([reg_out, softmax_out])  # concat to 1 single output
    model = Model(inputs=inp_shape, output=pred)
    model.compile(loss=softmax_mse_distance_loss, optimizer=optimizer)
    return model


def l2_normalize(x, axis):
    norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
    return K.maximum(x, K.epsilon()) / K.maximum(norm, K.epsilon())


def softmax_cosine_distance_loss(_y, y_pred):
    # split output into 2
    r_out = y_pred[:, :, :300]
    s_out = y_pred[:, :, -1]

    s = _y[:, :, -1]
    y = _y[:, :, :300]

    r_loss = cosine_distance_loss(y, r_out)
    s_out = K.clip(s_out, K.epsilon(), 1.)
    neg_s_out = K.clip(1 - s_out, K.epsilon(), 1.)
    s_loss = (s * (K.log(s_out))) + ((1 - s) * (K.log(neg_s_out)))
    s_loss = K.sum(-s_loss, axis=-1)
    # max of s_loss for each sample is ~16 (K.log(K.epsilon())) -> 0 < s_loss < 17
    # 0 < r_loss < 40
    loss = (r_loss + s_loss)
    return loss


def softmax_mse_distance_loss(_y, y_pred):
    # split output into 2
    r_out = y_pred[:, :, :300]
    s_out = y_pred[:, :, -1]

    s = _y[:, :, -1]
    y = _y[:, :, :300]

    r_loss = K.sum(mean_square_loss(y, r_out), axis=-1)
    s_out = K.clip(s_out, K.epsilon(), 1.)
    neg_s_out = K.clip(1 - s_out, K.epsilon(), 1.)
    s_loss = (s * (K.log(s_out))) + ((1 - s) * (K.log(neg_s_out)))
    s_loss = K.sum(-s_loss, axis=-1)
    # max of s_loss for each sample is ~16 (K.log(K.epsilon())) -> 0 < s_loss < 17
    loss = r_loss + s_loss
    return loss


def mean_square_loss(y, y_pred):
    return K.mean(K.square(y_pred - y), axis=-1)


def cosine_distance_loss(y, y_pred):
    # dot = K.sum((y * y_pred), axis=-1)
    # norm_y = l2_norm(y, axis=-1)
    # norm_y_pred = l2_norm(y_pred, axis=-1)

    # cos = dot / K.squeeze(norm_y * norm_y_pred, axis=2)
    # return K.sum(1 - cos, axis=-1)
    y = l2_normalize(y, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    cos = -K.mean(y * y_pred, axis=-1)
    return K.sum(1 - cos, axis=-1)
