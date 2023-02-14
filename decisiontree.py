import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import copy

#calculate the entropy value of a set of data according to the labels
def entropy(classes):
    classes = np.array(classes)
    n = len(classes)
    count0 = sum(classes == 0)
    count1 = sum(classes == 1)
    #all of the classes are the same
    #print(classes, count0, count1)
    if (count0 == n) or (count1 == n):
        return 0
    #compute the entropy value
    entropy_value = -(count0 / n) * np.log2(count0 / n) -(count1 / n) * np.log2(count1 / n)
    return entropy_value

def intrinsic(values):
    n = len(values)
    value_dict = Counter(values)
    intrinsic_value = sum([- (count / n) * np.log2(count / n)  
                           for count in value_dict.values()])
    return intrinsic_value

#split the data according to a feature and the threshold, return the infomation gain ratio,
# and the splitted datas
def entropy_gain(data, feature, c, classes):
    origin_entropy = entropy(classes)
    n = len(classes)
    classesL = classes[data[:, feature] < c]
    dataL = data[data[:, feature] < c]
    nL = len(classesL)
    classesR = classes[data[:, feature] >= c]
    dataR = data[data[:, feature] >= c]
    nR = len(classesR)
    entropyL = entropy(classesL)
    entropyR = entropy(classesR)
    info_gain = origin_entropy - (nL / n) * entropyL - (nR / n) * entropyR
    intrinsic_value = intrinsic(data[:, feature])
    info_gain_ratio = info_gain / intrinsic_value
    return info_gain_ratio, info_gain, dataL, dataR, classesL, classesR


def split(data, classes, preRule=''):
    if sum(classes == 0) > sum(classes == 1):
        majority_class = 0
    else:
        majority_class = 1
        
    if entropy(classes) == 0:
        return preRule + f': predict={majority_class}'
    
    best_gain_ratio = -1
    for feature in [0, 1]:
        if len(set(data[:, feature])) > 1:
            for c in set(data[:, feature]):
                if c == min(data[:, feature]):
                    continue
                info_gain_ratio, info_gain, dataL, dataR, classesL, classesR = entropy_gain(data, feature, c, classes)
                if info_gain_ratio > best_gain_ratio:
                    best_gain_ratio = info_gain_ratio
                    best_feature = feature
                    best_c = c
                    best_dataL = dataL
                    best_dataR = dataR
                    best_classesL = classesL
                    best_classesR = classesR
    if best_gain_ratio <= 0:
        return preRule + f': predict={majority_class}'
    else:
        return (split(best_dataL, best_classesL, preRule + f',x{best_feature + 1}<{best_c}'), 
                split(best_dataR, best_classesR, preRule + f',x{best_feature +1}>={best_c}'))


