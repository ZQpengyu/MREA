import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from stats_graph import stats_graph
# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#++++++++++++++++++++++++++++++++++++000000++++++++++++++++++++
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#tf.compat.v1.disable_eager_execution()
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#sess = tf.compat.v1.Session(config=config)
from keras.utils import multi_gpu_model
from keras.utils import plot_model
import data_helper
from keras.layers import Embedding, Input, Bidirectional, LSTM, Concatenate, Add, Dropout, Dense, \
    BatchNormalization, Lambda, Activation, multiply, concatenate, Flatten, add, Dot,Permute, GlobalAveragePooling1D,MaxPooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
from tensorflow.python.ops.nn import softmax



input_dim = data_helper.MAX_SEQUENCE_LENGTH
EMBDIM = data_helper.EMBDIM
embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')
model_data = data_helper.load_pickle('model_data.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0], EMBDIM, weights = [embedding_matrix], trainable=False)


def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in2_aligned = Dot(axes=1)([w_att_1, input_1])
    in1_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def self_align(input_1):
    attention = Dot(axes=-1)([input_1, input_1])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in1_aligned = add([input_1, in1_aligned])
    return in1_aligned

def multi_attn(input_1, input_2):
  input_2s = TimeDistributed(Dense(400))(input_2)
  attention = multiply([input_1, input_2s])
  attention = TimeDistributed(Dense(30, activation='tanh'))(attention)
  w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
  w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
  in2_aligned = Dot(axes=1)([w_att_1, input_1])
  in1_aligned = Dot(axes=1)([w_att_2, input_2])
  return in1_aligned, in2_aligned

def abs_attn(input_1, input_2):
  input_2s = TimeDistributed(Dense(400))(input_2)
  attention = Lambda(lambda x: K.abs(x[0]-x[1]))([input_1, input_2s])
  attention = TimeDistributed(Dense(30, activation='tanh'))(attention)
  w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
  w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
  in2_aligned = Dot(axes=1)([w_att_1, input_1])
  in1_aligned = Dot(axes=1)([w_att_2, input_2])
  return in1_aligned, in2_aligned
  
		     


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision

margin = 0.75
theta = lambda t: (K.sign(t)+1.)/2.

def loss(y_true, y_pred):
    return -(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(1-margin-y_pred)) * (y_true*K.log(y_pred + 1e-8) + (1-y_true)*K.log(1-y_pred+1e-8))

def matching(p,q):
    abs_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([p, q])
    #cos_diff = Lambda(lambda x: K.cos(x[0] - x[1]))([p, q])
    multi_diff = multiply([p, q])
    all_diff = concatenate([abs_diff, multi_diff])
    return all_diff

def base_model(input_shape):
    w_input = Input(shape = input_shape)
    c_input = Input(shape = input_shape)
    w_embedding = embedding_layer(w_input)
    c_embedding = embedding_layer(c_input)



    w_l = Bidirectional(LSTM(400,return_sequences='True', dropout=0.00003), merge_mode = 'sum')(w_embedding)
    c_l = Bidirectional(LSTM(400,return_sequences='True', dropout=0.00003), merge_mode = 'sum')(c_embedding)
    
    
    #w_attn = SelfAttention()([w_l, c_l])
    #c_attn = SelfAttention()([c_l, w_l])
    
    w_a1, c_a1 = align(w_l, c_l)
    w_a2, c_a2 = multi_attn(w_l, c_l)
    w_a3, c_a3 = abs_attn(w_l, c_l)
			
    
    
    
    w_l1 = add([w_l, w_a1])
    c_l1 = add([c_l, c_a1])
        
    w_l2 = add([w_l, w_a2])
    c_l2 = add([c_l, c_a2])
    

    
    #w_add, c_add = add_attn(w_l, c_l)
    w_l3 = add([w_l, w_a3])
    c_l3 = add([c_l, c_a3])
    w = concatenate([w_l1, w_l2, w_l3])
    c = concatenate([c_l1, c_l2, c_l3])    
    # w_align,w_lalign = align(w_embedding,w_l)
    # c_align, c_lalign = align(c_embedding, c_l)
    #wc = concatenate([wc_l, wc])#600
    
    #w = concatenate([w_embedding, w_l])#600
    #c = concatenate([c_embedding, c_l])#600

    #w = self_align(w)
    #c = self_align(c)

    
    #s = concatenate([w, c])


   # p = concatenate([w,c])


    model = Model([w_input, c_input],[w,c,w_l,c_l,w_l1,c_l1], name = 'base_model')
    model.summary()
    return model


def siamese_model():
    input_shape = (input_dim,)
    input_p1 = Input(shape = input_shape)
    input_p2 = Input(shape = input_shape)
    input_p3 = Input(shape = input_shape)
    input_p4 = Input(shape = input_shape)
    base_net = base_model(input_shape)

    pw,pc,pwl,pcl,pwl1,pcl1 = base_net([input_p1, input_p3])
    qw,qc,qwl,qcl,qwl1,qcl1 = base_net([input_p2, input_p4])
    
    
    pw_align, qw_align = align(pwl, qwl)
    pc_align, qc_align = align(pcl, qcl)
    pw_align = add([pwl, pw_align])
    pc_align = add([pcl, pc_align])
    qw_align = add([qwl, qw_align])
    qc_align = add([qcl, qc_align])
    
    
    
    pw_a1, qw_a1 = abs_attn(pwl, qwl)
    pc_a1, qc_a1 = abs_attn(pcl, qcl)
    
    pw_align1 = add([pwl, pw_a1])
    pc_align1 = add([pcl, pc_a1])
    qw_align1 = add([qwl, qw_a1])
    qc_align1 = add([qcl, qc_a1])
    
    pw_a2, qc_a2 = align(pwl, qcl)
    qw_a2, pc_a2 = align(qwl, pcl)

    
    
    
    
    
    pw = concatenate([pw,pw_align,pw_align1,pw_a2])
    pw = self_align(pw)
    qw = concatenate([qw,qw_align,qw_align1,qw_a2])
    qw = self_align(qw)
    pc = concatenate([pc,pc_align,pc_align1,pc_a2])
    pc = self_align(pc)
    qc = concatenate([qc,qc_align,qc_align1,qc_a2])
    qc = self_align(qc)
    
    
    
    
    
    
    
    p = concatenate([pw,pc])
    
    q = concatenate([qw,qc])
    
    ps = concatenate([pw_align1,pc_align1,pwl1,pcl1])
    qs = concatenate([qw_align1,qc_align1,qwl1,qcl1])
    ps,qs = align(ps,qs)
    


    
       
    
    p = GlobalMaxPooling1D()(p)
    q = GlobalMaxPooling1D()(q)
    ps = GlobalMaxPooling1D()(ps)
    qs = GlobalMaxPooling1D()(qs)
    #p_align1 = GlobalAveragePooling1D()(ps_align)
    #q_align1 = GlobalAveragePooling1D()(qs_align)

    
    all_diff1 = matching(p,q)
    all_diff2 = matching(ps,qs)
   # all_diff2 = matching(p_align1, q_align1)

    
    #all_diff = concatenate([p,q,all_diff1,all_diff2])
    all_diff = concatenate([p,q,all_diff1,all_diff2])

    


    # # print(all_diff)



    all_diff = Dropout(0.6)(all_diff)

    similarity = Dense(1200)(all_diff)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    #similarity = Flatten()(similarity)
    similarity = Dense(800)(similarity)
    similarity = Dropout(0.6)(similarity)
    similarity = Activation('relu')(similarity)
    #
    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
    model = Model([input_p1, input_p2, input_p3, input_p4], [similarity])
    # loss:binary_crossentropy;optimizer:adm,Adadelta
    model.summary()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', precision, recall, f1_score])

    return model

    
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1_score])
    #model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    #return model


def train():
    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_q3 = data['train_q3']
    train_q4 = data['train_q4']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_q3 = data['dev_q3']
    dev_q4 = data['dev_q4']
    dev_y = data['dev_label']

    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_q3 = data['test_q3']
    test_q4 = data['test_q4']
    test_y = data['test_label']
  
  
    #tensorboard_path = 'tensorboard'
    model = siamese_model()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    model_path = '2.best.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    #tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint,earlystopping, reduce_lr]

    history = model.fit([train_q1, train_q2, train_q3, train_q4], train_y,
                        batch_size=512,
                        epochs=200,
                        validation_data=([dev_q1, dev_q2, dev_q3, dev_q4], dev_y),
                        callbacks=callbackslist)
    '''
    ## Add graphs here
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])   
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss','train accuracy', 'val accuracy','train precision', 'val precision','train recall', 'val recall','train f1_score', 'val f1_score'], loc=3,
               bbox_to_anchor=(1.05,0),borderaxespad=0)
    pic = plt.gcf()
    pic.savefig ('pic.eps',format = 'eps',dpi=1000)
    plt.show()
    '''
    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2, test_q3, test_q4], test_y, verbose=1, batch_size=256)
    print("model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score))
    x = "model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score)
    model.save(model_path)

    with open('2.txt','a') as f:
      f.write(x)
      f.write('\n')

    #model = siamese_model()
    #model.load_weights(model_path)
    


if __name__ == '__main__':
  train()
 # train()


    


