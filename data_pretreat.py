''' 数据预处理 '''

''' 运行第一步 python data_pretreat.py --train_ True 第二步 python data_pretreat.py　'''
import argparse
import pandas as pd
import numpy as np
import os
from matplotlib import pylab as plt
# import 
from tqdm import tqdm
import time
plt.rcParams['font.sans-serif'] = ['SimHei']
parser = argparse.ArgumentParser()
parser.add_argument('--train_',type=bool ,default = False )
args = parser.parse_args()
train_ = args.train_

print (train_)
# bian_000 = -2 # train
if train_ :
    bian_000 = -2 # train
    shaep_0 = 7
    path_ = 'hy_round1_train_20200102/'
    save_path = 'data_train.npy'
else:
    bian_000 = -1 # test
    shaep_0 = 6
    path_ = 'hy_round1_testA_20200102/'
    save_path = 'data_test.npy'

def main(train_):
    data_ = []
    path_all = os.listdir(path_)
    path_all = sorted(path_all)
    pbar = tqdm( path_all )
    bain_hao_ = []
    for name in pbar:
        if '.csv' in name:
            pass
        else:
            continue
        temp_data = pd.read_csv( path_ + name )
        temp_data = np.array(temp_data)
        for dd in temp_data:
            if dd.shape[0] == shaep_0:  ### 7 yrain 6 test
                pass
            else:
                continue
            data_.append( dd[1:] )
            bain_hao_.append( dd[0] )
    if not train_:
        np.save('bianhao_.npy' , bain_hao_ )
    data_arr = np.array(data_)
    time_ti_jiao_0 = [dd.split(' ')[0] for dd in data_arr[:, bian_000]]
    time_ti_jiao_1 = [int(dd.split(' ')[1].split(':')[0])*3600 + \
                    int(dd.split(' ')[1].split(':')[1])*60 + \
                    int(dd.split(' ')[1].split(':')[2])   for dd in data_arr[:, bian_000] ]

    #时间分级
    time_1_ji_labels = []
    fen_zhong_ji_ = 15
    pbar = tqdm( time_ti_jiao_1 )
    jjjj_ = 0
    for dd in pbar:
        bbb_ = 0
        for ii in range( int(24*60/ fen_zhong_ji_ )  ):
            if dd>=ii*15 * 60 and dd<(ii + 1 )*15 * 60:
                time_1_ji_labels.append( ii )
                bbb_ = 1
                break
            else:
                pass

        jjjj_ =jjjj_+1
        if bbb_ == 0:
            print ('ooo')
            break


    pppp_ = [i for i in range(len(data_arr))]
    pbar = tqdm( pppp_ )
    key_oo_ = {'拖网':0,'围网':1,'刺网':2}

    data_end_ = []
    for i in pbar:
        pass
        if train_:
            data_tte_ = list(data_arr[i][:bian_000]) + \
            [int(time_ti_jiao_0[i])] + \
            [ time_1_ji_labels[i] ] + [key_oo_[data_arr[i][-1]]]
        else:
            data_tte_ = list(data_arr[i][:bian_000]) + \
            [int(time_ti_jiao_0[i])] + \
            [ time_1_ji_labels[i] ] 
        data_end_.append( data_tte_ )
    np.save(save_path,data_end_)
    print ('all is ok')

if __name__ == '__main__':
    main(train_)














