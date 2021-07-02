import logging
import math
import os
import time
import csv
import sys
import numpy as np
import tensorflow as tf
from net import createModel, defineExperimentPaths
from keras.callbacks import (EarlyStopping, LearningRateScheduler)
from sklearn.metrics import (accuracy_score)
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示worning
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

vec_filepath = './model/SM424.npz' #./model/SM580.npz
basic_path = os.getcwd()+'/'+'output/'
np.random.seed(2021)
def get_vec(filepath):
    vec = []
    label = []

    vec_file = np.load(vec_filepath, allow_pickle=True)
    #feat_file = np.load('Phys5.npz', allow_pickle=True)
    #feat_file2 = np.load('pssm.npz', allow_pickle=True)

    for i in vec_file.keys():
        if 'Matrix' in i:
            vec.append(vec_file[i])
            label.append(np.array([1, 0, 0, 0]))
        if 'Intermembrane_Space' in i:
            vec.append(vec_file[i])
            label.append(np.array([0, 1, 0, 0]))
        if 'Inner_Membrane' in i:
            vec.append(vec_file[i])
            label.append(np.array([0, 0, 1, 0]))
        if 'Outer_Membrane' in i:
            vec.append(vec_file[i])
            label.append(np.array([0, 0, 0, 1]))

    vec = np.array(vec).reshape(424, 1, 1024)

    label = np.array(label)

    #label = label.reshape(424, 1, 4)
    print(label.shape)
    return vec, label


def trans_to(y, label):
    a = []
    for i in y:
        if i == label:
            i = 1
        else:
            i = 0
        a.append(i)
    return np.array(a)


def dataset_split(vec, label, random_state):
    ten_fold = []
    label = np.argmax(label, axis=-1)
    sss = StratifiedKFold(n_splits=10, random_state=79, shuffle=True)  # 5  162
    for train_index, test_index in sss.split(vec, label):
        ten_fold.append(test_index)
    init = 0
    for _ in range(0, 10):
        train_index = []
        val_num = 9 - init
        test_index = list(ten_fold[init])
        val_index = list(ten_fold[val_num])
        for i in range(0, 10):
            if i != init and i != val_num:
                train_index += list(ten_fold[i])
        yield (train_index, val_index, test_index)
        init += 1


def GCC_calcu(y_pre, y_true, K=4):
    M_matrix = np.zeros((4, 4))
    a = np.zeros((4, 1))
    b = np.zeros((4, 1))
    e = np.zeros((4, 4))
    for i in range(len(y_true)):
        M_matrix[y_true[i]][y_pre[i]] += 1

    for i in range(0, 4):
        a[i] = M_matrix.sum(axis=1)[i]
        b[i] = M_matrix.sum(axis=0)[i]

    for i in range(0, 4):
        for j in range(0, 4):
            e[i][j] = a[i]*b[j]/len(y_true)
    gcc = 0.0
    for i in range(0, 4):
        for j in range(0, 4):
            if e[i][j] == 0:
                break
            else:
                gcc += (M_matrix[i][j]-e[i][j])**2/e[i][j]
    GCC = (gcc/(len(y_true)*(K-1)))**0.5
    return GCC


def main(ex_time):
    result_mcc = open("162.csv", 'a')

    MCC = []
    GCC = []
    i = 1
    Acc_M, Acc_T, Acc_I, Acc_O = [], [], [], []
    MCC_M, MCC_T, MCC_I, MCC_O = [], [], [], []

    batchSize = 8
    maxEpochs = 100
    random_state = int(ex_time)

    vec, label = get_vec(vec_filepath)
    gen = dataset_split(vec, label, random_state)
    for train_index, test_index, val_index in gen:
        print(train_index)
        print(test_index)
        train_X = vec[train_index]
        train_y = label[train_index]
        test_X = vec[test_index]
        test_y = label[test_index]
        val_X = vec[val_index]
        val_y = label[val_index]

        logging.debug("Loading network/training configuration...")
        basic_path2 = basic_path + ex_time
        [MODEL_PATH, CHECKPOINT_PATH, LOG_PATH, RESULT_PATH] = defineExperimentPaths(basic_path2,
                                                                                     '/'+str(i))
        model = createModel()
        logging.debug("Model summary ... ")
        model.count_params()
        model.summary()

        model.compile(optimizer='adam',
                      loss={'ss_output': 'categorical_crossentropy'}, metrics=['accuracy'])
        logging.debug("Running training...")

        def step_decay(epoch):
            initial_lrate = 0.001
            drop = 0.5
            epochs_drop = 6.0
            lrate = initial_lrate * \
                math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            print(lrate)
            return lrate

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5,
                          verbose=2, mode='auto'),
            # ModelCheckpoint(checkpoint_weight,
            #                 monitor='val_loss',
            #                 verbose=1,
            #                 save_best_only=True,
            #                 mode='auto',
            #                 period=1),
            LearningRateScheduler(step_decay),
        ]
        startTime = time.time()

        mean_fpr = np.linspace(0, 1, 100)  # 返回一个从0-1 的等差数列 个数为100个
        history = model.fit(
            {'sequence_input': train_X},
            {'ss_output': train_y},
            epochs=maxEpochs,
            batch_size=batchSize,
            callbacks=callbacks,
            verbose=1,
            validation_data=(
                {'sequence_input': val_X},
                {'ss_output': val_y}),
            shuffle=False)
        endTime = time.time()
        logging.debug("Saving final model...")
        model.save(os.path.join(MODEL_PATH, 'model.h5'), overwrite=True)
        json_string = model.to_json()
        with open(os.path.join(MODEL_PATH, 'model.json'), 'w') as f:
            f.write(json_string)
        logging.debug("make prediction")
        ss_y_hat_test = model.predict(
            {'sequence_input': test_X})
        
        y_true = test_y
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(ss_y_hat_test, axis=-1)

        gcc = GCC_calcu(y_pred, y_true)

        y_true_M, y_pred_M = trans_to(y_true, 0), trans_to(y_pred, 0)
        y_true_T, y_pred_T = trans_to(y_true, 1), trans_to(y_pred, 1)
        y_true_I, y_pred_I = trans_to(y_true, 2), trans_to(y_pred, 2)
        y_true_O, y_pred_O = trans_to(y_true, 3), trans_to(y_pred, 3)

        acc_m = accuracy_score(y_true_M, y_pred_M)
        mcc_m = matthews_corrcoef(y_true_M, y_pred_M)
        acc_t = accuracy_score(y_true_T, y_pred_T)
        mcc_t = matthews_corrcoef(y_true_T, y_pred_T)
        acc_i = accuracy_score(y_true_I, y_pred_I)
        mcc_i = matthews_corrcoef(y_true_I, y_pred_I)
        acc_o = accuracy_score(y_true_O, y_pred_O)
        mcc_o = matthews_corrcoef(y_true_O, y_pred_O)

        GCC.append(gcc)
        Acc_M.append(acc_m)
        MCC_M.append(mcc_m)
        Acc_T.append(acc_t)
        MCC_T.append(mcc_t)
        Acc_I.append(acc_i)
        MCC_I.append(mcc_i)
        Acc_O.append(acc_o)
        MCC_O.append(mcc_o)
        print("-------------------------------------------------" + str(
            i))

        print("第 %d 折 acid ACC_M: %.4f \t acid MCC_M: %.4f " %
              (i, acc_m, mcc_m))
        print("第 %d 折 acid ACC_T: %.4f \t acid MCC_T: %.4f " %
              (i, acc_t, mcc_t))
        print("第 %d 折 acid ACC_I: %.4f \t acid MCC_I: %.4f " %
              (i, acc_i, mcc_i))
        print("第 %d 折 acid ACC_O: %.4f \t acid MCC_O: %.4f " %
              (i, acc_o, mcc_o))
        print("\n")
        print("第 %d 折 acid GCC: %.4f " % (i, gcc))
        i = i + 1

    print("acid ACC_M: %.4f \t acid MCC_M: %.4f " %
          (np.mean(Acc_M), np.mean(MCC_M)))
    print("acid ACC_T: %.4f \t acid MCC_T: %.4f " %
          (np.mean(Acc_T), np.mean(MCC_T)))
    print("acid ACC_I: %.4f \t acid MCC_I: %.4f " %
          (np.mean(Acc_I), np.mean(MCC_I)))
    print("acid ACC_O: %.4f \t acid MCC_O: %.4f " %
          (np.mean(Acc_O), np.mean(MCC_O)))
    print("acid GCC: %.4f " % (np.mean(GCC)))

    MCC.append(np.mean(MCC_M))
    MCC.append(np.mean(MCC_T))
    MCC.append(np.mean(MCC_I))
    MCC.append(np.mean(MCC_O))
    MCC.append(random_state)
    MCC.append(np.mean(GCC))

    csv.writer(result_mcc).writerow(MCC)

    result_mcc.close()


if __name__ == "__main__":
        #i = sys.argv[1]
        main(str(1))
