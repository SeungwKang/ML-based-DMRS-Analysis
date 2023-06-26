import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score ,confusion_matrix
from keras.models import Sequential
from keras import optimizers, losses, callbacks, regularizers
from keras.layers import Dense, Activation, Dropout, InputLayer, BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
import time


SNRs = ['23.37', '9.56', '-2.51','-2.74','-2.81','-2.99','-3.13','-3.42','-3.7','-4.11','-4.5','-4.66']
dirs = ['origin','power_averaging','sequence_scaling','PCA']


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
                                average='macro')
    result.append(precision)

    #재현율
    recall = recall_score(y_test, pred,
                         average='macro')
    result.append(precision)

    #F1 score
    f1 = f1_score(y_test, pred,
                 average='macro')
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


#initialization of early stopper
early_stopper = callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', min_delta=0, patience=20, verbose=1)

datapath = 'D:/DMRS/'
result_path = path + 'results/'



for dir in range(0,4):
    eval_logs = []
    for snr in range(0,12):
        #model save initialization
        checkpoint_filepath = result_path+'results_DNN/'+dirs[dir]+'/fig/'+str(snr) + '_accuracy_'+dirs[dir]+'.hdf5'
        save_best = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath, monitor='val_sparse_categorical_accuracy', verbose=1, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch', options=None)

        #data second pre-processing
        data = pd.read_csv(datapath+dirs[dir]+'/'+str(snr)+'_SNR.csv', encoding='CP949')
        Label = data['index'].to_numpy()
        X = data.drop(['index'],axis='columns')
        col_list  = X.columns
        X = X.to_numpy()
        print('X_shape : ', X.shape,'  Y_shape : ', Label.shape)

        #train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Label, stratify=Label, test_size = 0.2, random_state = 33)

        #save test data indivisually
        result_csv = pd.DataFrame(X_test, columns=col_list)
        result_csv['index'] = Y_test.astype(int)
        print(result_csv.head())
        print(result_csv.info())
        result_csv.to_csv(datapath+dirs[dir]+'/temp/'+str(snr)+'_SNR.csv', index=False)

        #model
        #regularizer = None#regularizers.l2(0.01)
        model = Sequential()
        model.add(InputLayer(input_shape =(X_train.shape[1],)))
        model.add(Dense(512))# kernel_regularizer=regularizer))
        model.add(Activation('relu'))
        model.add(Dense(512))#kernel_regularizer=regularizer))
        model.add(Activation('relu'))
        model.add(Dense(512))#kernel_regularizer=regularizer))
        model.add(Activation('relu'))
        model.add(Dense(512))#kernel_regularizer=regularizer))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(),loss=losses.SparseCategoricalCrossentropy(), metrics=['sparse_categorical_accuracy'])
        start = time.time()
        history = model.fit(X_train,Y_train, epochs=1000, validation_split=0.2,verbose=2, batch_size=512, callbacks=[early_stopper, save_best])
        trainingTime = time.time() - start
        print("time :", trainingTime)

        #PLOT
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title(SNRs[snr] + ' accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        img_name = str(snr) + '_accuracy_'+dirs[dir]+'.png'
        plt.savefig(result_path+'results_DNN/'+dirs[dir]+'/fig/'+img_name,dpi=300)#DNN*************
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
    savelog_df.to_csv(result_path+'results_DNN/'+dirs[dir]+'/ResultLog_'+dirs[dir]+'_DNN.csv' )
