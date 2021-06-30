"""
Liveness evaluation functions.
"""

import os 
import glob
import numpy as np
from scipy import interp  
from sklearn.metrics import roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt  


recall_th = [0.1, 0.05, 0.01, 0.001, 0.0001]
tpr_th = [0.8]
hack_th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def plot_show(y,prob):
    fpr, tpr, thresholds = roc_curve(y, prob, pos_label=1) 
    mean_tpr_list = []
    for i in range(len(recall_th)):
        mean_tpr_list.append(interp(recall_th[i], fpr, tpr))
    for i in range(len(recall_th)):
        print(i, 'tpr@fpr:', recall_th[i], mean_tpr_list[i]) 
    
    for i in range(len(hack_th)):
        print(hack_th[i], 'fpr@tpr:', interp(hack_th[i], thresholds, tpr, period=1), interp(hack_th[i], thresholds, fpr, period=1), hack_th[i] ) 

    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = thresholds[right_index]
    #best_th = 0.2
    err = fpr[right_index]
    mean_accuracy = 1-err

    return recall_th, mean_tpr_list

def _get_max_safety(num_split):
    max_hack = '0.0'
    for score_str in num_split[2:5]:
        if float(score_str) >= float(max_hack):
            max_hack = score_str
    return max_hack
    

def get_scores(path_txt, isreverse = False):
    fid = open(path_txt, 'r', encoding='utf-8')
    lines = fid.readlines() 
    print('Samples num:', len(lines))
    fid.close()

    test_scores = []#np.zeros([len(lines)])
    test_labels = []#np.zeros([len(lines)])

    for i in range(len(lines)):        
        line = lines[i]
        num_split = line.strip().split(' ')
        if len(num_split)==2:
            image_name = num_split[0]
            # label = image_name[-1]
            score_str = num_split[-1]
        elif len(num_split)>=3:
            # num_split = line.strip().split(']')[0].strip().split(' ')
            image_name = num_split[0]
            label = num_split[1]
            score_str = num_split[-1]
        else:
            print('error score content!')
            exit(1)
        '''
        if 'real' in image_name: 
            test_labels += [1]
            test_scores += [float(score_str)]
        elif True:
            test_labels += [0]
            test_scores += [float(score_str)]
        else:
            pass
        '''
        if label == "0":
            test_labels += [0]
            test_scores += [float(score_str)]
        elif label == "1":
            # 如果这些词在图片名中表示该图片为正例
            test_labels += [1]
            test_scores += [float(score_str)]
        else:
            print('error image name!')
            exit(1)


    print('Total Test Samples:', len(test_scores))
    test_labels = np.array(test_labels)
    test_scores = np.array(test_scores)    
    
    print('Good Test Samples:', np.sum(test_labels==1))
    print('Bad Test Samples:', np.sum(test_labels==0))
    return test_labels, test_scores


def eval_roc(score_file):
    print('evaling score file:', score_file)
    labels, scores = get_scores(score_file)
    recall_th_list, mean_tpr_list = plot_show(labels, scores)

    performance_str = ''
    for _recal_th, _mean_tpr in zip(recall_th_list, mean_tpr_list):
        performance_str += '%s=%.5f '%(str(_recal_th), _mean_tpr)
    return performance_str 



if __name__ == '__main__':
    
    score_file = "/home/projects/list/test_result_transformer/checkpoint_20210524_epoch_2_cbsr_mas_v6_hifi-mask-test_bbox.txt"
    eval_roc(score_file)
