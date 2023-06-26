# Normal Data load(Normal: 원본 DMRS데이터에 Normalization 적용)========================
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score ,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # 결정트리 분류기
from sklearn.ensemble import RandomForestClassifier # Bagging model-Random Forest 분류기
from lightgbm import LGBMClassifier # Boosting model-LightGBM 분류기(먼저 설치 필요)
from sklearn.ensemble import VotingClassifier # Voting 분류기
import time
import pickle


## 정확도, 정밀도, 재현률, F1 스코어, ROC_AUC값 출력을 위한 함수 생성========================================================
def get_clf_eval(model_name, y_test, pred=None, pred_proba=None):
    len_y = len(y_test)
    print("Number of test data: {}".format(len_y))
    result = []
    result.append(model_name)

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

    return result
#======================================================================================================================

datafolder = "D:/종석/연구실/DMRS_data/[20221115]DMRS/"
resultfolder = "D:/종석/연구실/DMRS결과/1.csv"

data_format = ["Origin", "PCA", "power_averaging", "sequence_scaling"]
for pre_data in data_format :
    for n in range(12):
        folder =datafolder+pre_data+'/'+str(n)+"_SNR"
        path = folder+".csv"
        print('Training {}...'.format(path))
        df_DMRS = pd.read_csv(path)
        #print(df_DMRS.head())
        # 학습데이터(x), 정답데이터(y) 구분
        x = df_DMRS.iloc[:,:-1]
        #print("학습데이터:\n", x.head())
        y = df_DMRS['index']
        #print("\n정답데이터:\n", y.head())
        #======================================================================================

        # 학습, 검증, 테스트데이터 분할============================================================
        x = x.to_numpy()
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                          train_size=0.8,
                                                          shuffle=True,
                                                          stratify=y,
                                                          random_state=11) # 학습, 테스트 데이터 분할

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          train_size=0.8,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=11) # 학습, 평가 데이터 분할
        #========================================================================================


        # 모델 생성======================================================================================================

        # DecisionTree 분류기
        dt_clf = DecisionTreeClassifier(max_depth=24, random_state=11)

        # Random Forest 분류기
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=11)

        # LightGBM 분류기 생성
        lgbm_clf = LGBMClassifier(n_estimators=1000, learning_rate=0.1, objective='multi:softmax', random_state=11)

        # Stacking 분류기 생성
        stack_clf = LGBMClassifier(n_estimators=1000, learning_rate=0.1, objective='multi:softmax', random_state=11)
        #=================================================================================================================


        # 모델 학습======================================================================================================
        # 결정트리 학습시작
        print("Decision Tree Train Start")
        dtstart_time = time.time()
        dt_clf.fit(x_train, y_train)
        df_model = pickle.dumps(dt_clf)
        dt_traintime = time.time()-dtstart_time
        #print("Time:{:.4f} sec".format((time.time()-dtstart_time)))
        print("Decision Tree Train End")
        # 결정트리 학습 끝

        # RandomForest 학습시작
        print("\nRandomForest Train Start")
        rfstart_time = time.time()
        rf_clf.fit(x_train, y_train)
        rf_traintime = time.time()-rfstart_time
        #print("Time:{:.4f} sec".format((time.time()-rfstart_time)))
        print("RandomForest Train End")
        # RandomForest 학습 끝

        # LightGBM 학습시작
        print("\nLightGBM Train Start")
        lgbmstart_time = time.time()
        lgbm_clf.fit(x_train, y_train, eval_metric='multi_logloss', eval_set=[(x_val, y_val)], verbose=False)
        lgbm_traintime = time.time()-lgbmstart_time
        #print("Time:{:.4f} sec".format((time.time()-lgbmstart_time)))
        print("LightGBM Train End")
        # LightGBM 학습 끝
        #=================================================================================================================

        # 분류기 평가==============================================
        eval_logs = []
        # Decision Tree 평가
        dt_preds = dt_clf.predict(x_test)
        dt_proba = dt_clf.predict_proba(x_test)
        model_name = 'DecisionTree'
        #print('\nDecision Tree 분류기 평가')
        eval_log = get_clf_eval(model_name, y_test, dt_preds, dt_proba)
        eval_log.append(dt_traintime)
        eval_logs.append(eval_log)

        # Random Forest 평가
        rf_preds = rf_clf.predict(x_test)
        rf_proba = rf_clf.predict_proba(x_test)
        model_name = 'RandomForest'
        #print('\nRandom Forest 분류기 평가')
        eval_log = get_clf_eval(model_name, y_test, rf_preds, rf_proba)
        eval_log.append(rf_traintime)
        eval_logs.append(eval_log)

        # LightGBM 평가
        lgbm_preds = lgbm_clf.predict(x_test)
        lgbm_proba = lgbm_clf.predict_proba(x_test)
        model_name = 'LightGBM'
        #print('\nLightGBM 분류기 평가:')
        eval_log = get_clf_eval(model_name, y_test, lgbm_preds, lgbm_proba)
        eval_log.append(lgbm_traintime)
        eval_logs.append(eval_log)

        # Stacking model 학습 및 평가 ========================================
        # 이전 Layer model들의 예측값을 Final Classifier의 학습데이터로 만드는 과정
        layer_preds = np.array([dt_preds, rf_preds, lgbm_preds])
        layer_preds = np.transpose(layer_preds)

        # Stacking model 학습시작
        print("\nStacking model Train Start")
        stackstart_time = time.time()
        stack_clf.fit(layer_preds, y_test)
        stack_traintime = (time.time()-stackstart_time)
        print("Stacking model Train End")
        # Stacking model 학습 끝

        # Stacking model 평가
        stack_preds = stack_clf.predict(layer_preds)
        stack_proba = stack_clf.predict_proba(layer_preds)
        model_name = 'Stacking'
        #print('\nStacking model 분류기 평가:')
        eval_log = get_clf_eval(model_name, y_test, stack_preds, stack_proba)
        eval_log.append(stack_traintime)
        eval_logs.append(eval_log)
        #======================================================================

        # Log 저장 ========================================================================================
        print("평가용 로그:",eval_logs)

        # DT: Decision Tree, RF: Random Forest, LG: LightGBM, VT: Voting model, ST: Stacking model
        columns = ['ML_model','Accuracy', 'Precision', 'Recall', 'F1_score', 'ROC_AUC', 'Training Time']

        savelog_df = pd.DataFrame(data=eval_logs, columns=columns)
        print(savelog_df)
        #==================================================================================================

        # Log CSV 저장=================================================================================
        savelog_df.to_csv(resultfolder)
        #==============================================================================================
