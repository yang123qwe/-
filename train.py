''' 训练代码　预测得到结果 '''
import pandas as pd
import numpy as np
import os
from matplotlib import pylab as plt
from tqdm import tqdm
import time
plt.rcParams['font.sans-serif'] = ['SimHei']

# !pip install lightgbm
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter

def one_hot(train_test_ti ,train = True):
    encoder = LabelEncoder()  
    ocean = encoder.fit_transform( train_test_ti )
    enc = OneHotEncoder(categories='auto')
    ti_jiao_=enc.fit_transform( ocean.reshape(-1,1) )
    ti_jiao_=ti_jiao_.toarray()
    if train:
        train_ti_jiao_ = ti_jiao_[:data.shape[0] ,: ]
        test_ti_jiao_ = ti_jiao_[ data.shape[0] : ,: ]
        return train_ti_jiao_ , test_ti_jiao_
    else:
        return ti_jiao_ , 0
    
def bain_zhu(data , train=True):
    
    X_point_max_ = max(data[:,0])
    X_point_min_ = min(data[:,0])

    Y_point_max_ = max(data[:,1])
    Y_point_min_ = min(data[:,1])

    su_du_max_ = max(data[:,2]) 
    su_du_min_ = min(data[:,2])

    fang_xiang_max_ = max(data[:,3]) 
    fang_xiang_min_ = min(data[:,3])
    
    # -1~1
    gui_yi_X = (2 * ((data[:,0] - X_point_min_) / (X_point_max_ - X_point_min_)) - 1)
    gui_yi_X = gui_yi_X.reshape(-1, 1)
    gui_yi_Y = (2 * ((data[:,1] - Y_point_min_) / (Y_point_max_ - Y_point_min_)) - 1)
    gui_yi_Y = gui_yi_Y.reshape(-1, 1)
    # 0~1
    gui_yi_su = ((data[:,2] - su_du_min_) / (su_du_max_ - su_du_min_)) 
    gui_fang_xiang = ((data[:,3] - fang_xiang_min_) / (fang_xiang_max_ - fang_xiang_min_)) 
    gui_yi_su = gui_yi_su.reshape(-1, 1)
    gui_fang_xiang = gui_fang_xiang.reshape(-1, 1)
    #one-hot
    train_ti_jiao_ , test_ti_jiao_ = one_hot(train_test_ti)

    train_time , test_time = one_hot(train_test_time)

    if train:
        labels_train_ , _ = one_hot( data[:,6].astype(int) ,train=False) 
        TR_data = np.concatenate([gui_yi_X,gui_yi_Y,
                           gui_yi_su,gui_fang_xiang,
                           train_ti_jiao_ ,train_time ,
                           labels_train_] , axis=1)
    else:
        TR_data = np.concatenate([gui_yi_X,gui_yi_Y,
                   gui_yi_su,gui_fang_xiang,
                   test_ti_jiao_ ,test_time] , axis=1)
    return TR_data


data = np.load('data_train.npy')
test_ = np.load('data_test.npy')
train_test_ti = np.concatenate( [data[:,4].astype(int) ,test_[:,4].astype(int)] )
train_test_time = np.concatenate( [data[:,5].astype(int) ,test_[:,5].astype(int)] )

train_data_all_ = bain_zhu(data , train=True)
test_data_all_ = bain_zhu(test_ , train=False)
X_Y_array  = shuffle( train_data_all_  )
X_all_ = X_Y_array[:,:-3]
Y_all_ = X_Y_array[:,-3:]

###### model
params = {
    'n_estimators': 6000,
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 3,
    'early_stopping_rounds': 100,
    'num_thread' : 30
}

models = []
oof = np.zeros((len( X_all_ ), 3))
pred = np.zeros((len(test_data_all_),3))

fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for index, (train_idx, val_idx) in enumerate(fold.split(X_all_, Y_all_.argmax(-1))):
    train_set = lgb.Dataset( X_all_[train_idx] , Y_all_.argmax(-1)[train_idx])
    
    val_set = lgb.Dataset( X_all_[val_idx]  , Y_all_.argmax(-1)[val_idx] )
    
    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
    models.append(model)
    val_pred = model.predict( X_all_[val_idx] )
    oof[val_idx] = val_pred
    val_y =  Y_all_.argmax(-1)[val_idx]
    val_pred = np.argmax(val_pred, axis=1)
    
    print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
    # 0.8695539641133697
    # 0.8866211724839532

    test_pred = model.predict( test_data_all_ )
    pred += test_pred/5

bian_hao_ = np.load('bianhao_.npy')
key_oo_ = {'拖网':0,'围网':1,'刺网':2}
ind_ = {}
for kk in list(key_oo_.keys()):
    ind_[ key_oo_[kk] ] = kk
pred_copy = np.argmax(pred, axis=1)

result_num_dic_ = {}
for ii in range( pred_copy.shape[0] ):
    if bian_hao_[ii] in result_num_dic_:
        result_num_dic_[ bian_hao_[ii] ].append( pred_copy[ii] )
    else:
        result_num_dic_[ bian_hao_[ii] ] = [pred_copy[ii]]

re_keys = list(result_num_dic_.keys())
result_num_mean_ = []
result_max_ind = []
result_fen_bu_dic_ = {}
for kk_ in re_keys:
    
    con_0 = Counter(result_num_dic_[kk_])
    X_0_c = list(con_0.keys())
    Y_0_c = [ con_0[dd] for dd in X_0_c]

    ind_00 = X_0_c[Y_0_c.index(max(Y_0_c))]
    result_num_mean_.append( [kk_ , np.mean(result_num_dic_[kk_]) ] )
    result_max_ind.append( [kk_ , ind_[ ind_00 ] ] )
    result_fen_bu_dic_[kk_] = dict(con_0)
#     break

TT = pd.DataFrame(result_max_ind)
TT.to_csv('result.csv' ,header=False , index = False , encoding = 'utf-8')
print(TT[1].value_counts(1))






























