import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score ,confusion_matrix
from keras.models import Sequential
from keras import optimizers, losses, callbacks
from keras.layers import Dense, Activation, Dropout, InputLayer, BatchNormalization, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math


SNRs = ['23.37', '9.56', '-2.51','-2.74','-2.81','-2.99','-3.13','-3.42','-3.7','-4.11','-4.5','-4.66']

dataType = ['origin','power_averaging','sequence_scaling']


## 정확도, 정밀도, 재현률, F1 스코어, ROC_AUC값, 훈련시간 출력을 위한 함수 생성========================================================
def get_clf_eval(y_test, pred=None, pred_proba=None, trainTime = None, epochs = None):
    len_y = len(y_test)
    print("Number of test data: {}".format(len_y))
    result = []

    confusion = confusion_matrix(y_test, pred)
    #confusion = confusion/len_y*100
    #정확도
    accuracy = accuracy_score(y_test, pred)
    result.append(accuracy)

    #정밀도
    precision = precision_score(y_test, pred,
                                average='micro')
    result.append(precision)

    #재현율
    recall = recall_score(y_test, pred,
                         average='micro')
    result.append(precision)

    #F1 score
    f1 = f1_score(y_test, pred,
                 average='micro')
    result.append(f1)

    # ROC_AUC
    roc_auc = roc_auc_score(y_test, pred_proba,
                            multi_class="ovr",
                            average='macro')
    result.append(roc_auc)

    result.append(trainTime)

    result.append(epochs)

    return result
#======================================================================================================================

#DataLoader
class Dataloader(Sequence):

    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

		# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

#initialization of early stopper
early_stopper = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0, patience=20, verbose=1)

path = 'D:/DMRS/'

for idx, dt in enumerate(dataType):
    eval_logs = []
    for snr in range(0,12):
        data_path = path + dt
        result_path = path + 'results/results_CNN1D/'+dt
        size_per_seq = 144
        #model save initialization
        checkpoint_filepath = result_path+'/fig/'+str(snr) + '_accuracy.hdf5'#DNN************
        save_best = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch', options=None)

        #data second pre-processing
        data = pd.read_csv(data_path+'/'+str(snr)+'_SNR.csv', encoding='CP949')
        XAcol = ['a' + str(i) for i in range(0,144)]
        XBcol = ['b' + str(i) for i in range(0,144)]

        XA = data[XAcol].to_numpy()
        XB = data[XBcol].to_numpy()
        print(XA.shape, XB.shape)

        Label = data['index'].to_numpy()

        XA = XA.reshape(-1, 1, size_per_seq)
        XB = XB.reshape(-1, 1, size_per_seq)

        X = np.concatenate([XA,XB],axis=1)

        #train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Label, stratify=Label, test_size = 0.2, random_state = 33)


        #save test data indivisually
        X_test_temp = X_test.reshape(-1, size_per_seq*2)
        col_list  = ['a' + str(i) for i in range(0,144)] + ['b' + str(i) for i in range(0,144)]
        result_csv = pd.DataFrame(X_test_temp, columns=col_list)
        result_csv['index'] = Y_test.astype(int)
        print(result_csv.head())
        print(result_csv.info())
        result_csv.to_csv(data_path+'/CNNtemp/'+str(snr)+'_SNR.csv', index=False)


        #train validation split
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size = 0.2, random_state = 33)
        train_loader = Dataloader(X_train,Y_train, 512, shuffle=True)
        val_loader = Dataloader(X_val,Y_val, 512)
        print(X_train.shape)

        #model
        model = Sequential()
        model.add(Conv1D(input_shape = (X_train.shape[1], X_train.shape[2]), filters = 4, kernel_size = 10, padding = 'same', data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size = 2, data_format="channels_first"))
        model.add(Conv1D(filters = 8, kernel_size = 5, padding = 'same',data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(Conv1D(filters = 16, kernel_size = 5, padding = 'same',data_format="channels_first"))
        model.add(Activation('relu'))
        model.add(Flatten())

        model.add(Dense(256, activation='relu'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(),loss=losses.SparseCategoricalCrossentropy(), metrics=['sparse_categorical_accuracy'])
        start = time.time()
        history = model.fit(train_loader, epochs=1000, validation_data=val_loader,verbose=2, callbacks=[early_stopper, save_best])
        trainingTime = time.time() - start
        print("time :", trainingTime)

        #PLOT
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title(SNRs[snr] + ' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        img_name = str(snr) + '_accuracy.png'
        plt.savefig(result_path+'/fig/'+img_name,dpi=300)#DNN*************
        plt.clf()
        #end PLOT

        #model evaluation
        model.load_weights(checkpoint_filepath)
        prediction = np.argmax(model.predict(X_test), axis=-1)
        prediction_proba = model.predict(X_test)
        print("EPOCHS : ", len(history.history['sparse_categorical_accuracy']))
        results = get_clf_eval(Y_test, prediction,prediction_proba,trainingTime, len(history.history['sparse_categorical_accuracy']))
        print("results : ", results)
        eval_logs.append(results)

    #save results
    columns_results = ['Accuracy', 'Precision', 'Recall', 'F1_score', 'ROC_AUC', 'Training Time', 'EPOCHS']
    savelog_df = pd.DataFrame(data=eval_logs, columns=columns_results)
    savelog_df.to_csv(result_path+'/ResultLog_CNN.csv' )
