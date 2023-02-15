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
    #all of the classes are same
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

row_data4 = open('D3leaves.txt').read().split('\n')[:-1]
data4 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data4])
classes4 = np.array([int(line.split()[2]) for line in row_data4])

tree4 = split(data4, classes4)
print(tree4) 

#3
row_data = open('Druns.txt').read().split('\n')[:-1]
data3 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data])
classes3 = np.array([int(line.split()[2]) for line in row_data])

results = []
zero_info_gain_rate = []
for feature in [0, 1]:
    if len(set(data3[:, feature])) > 1:
        for c in set(data3[:, feature]):
            if c == min(data3[:, feature]):
                continue
            info_gain_ratio, info_gain, dataL, dataR, classesL, classesR = entropy_gain(data3, feature, c, classes3)
            info_gain_ratio = info_gain_ratio.round(5)
            entropyL = entropy(classesL)
            entropyR = entropy(classesR)
            if (entropyL + entropyR) == 0:
                zero_info_gain_rate.append([f'x{feature}', c, info_gain])
            results.append([f'x{feature + 1}', c, info_gain_ratio])
            
df3 = pd.DataFrame(results)
df3.columns = ['split feature', 'threshold', 'information gain ratio']
df3

#4
row_data4 = open('D3leaves.txt').read().split('\n')[:-1]
data4 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data4])
classes4 = np.array([int(line.split()[2]) for line in row_data4])

tree4 = split(data4, classes4)
print(tree4)
# display the tree in a formatted string
#5
#(1)
def format_tree(tree, i=0):
    if type(tree) == str:
        level = len(tree.split(',')) - 1
        print('\t' * level + '\u0332'.join(tree.split(':')[-1][1:]))
    else:
        rule0 = get_final_rule(tree[0])
        rule1 = get_final_rule(tree[1])
        decision_boundries.append(rule0.split(':')[0].split(',')[i+1].split('<'))
        print('\t' * i + rule0.split(':')[0].split(',')[i+1])
        format_tree(tree[0], i+1)
        print('\t' * i + rule1.split(':')[0].split(',')[i+1])
        format_tree(tree[1], i+1)

def get_final_rule(tree):
    if type(tree) == str:
        return tree
    else:
        return get_final_rule(tree[0])

row_data5_1 = open('D1.txt').read().split('\n')[:-1]
data5_1 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data5_1])
classes5_1 = np.array([int(line.split()[2]) for line in row_data5_1])

tree5_1 = split(data5_1, classes5_1)
# The tree of D1.txt is:
global decision_boundries
decision_boundries = []
format_tree(tree5_1)

db1 = copy.deepcopy(decision_boundries)
#(3)
row_data5_2 = open('D2.txt').read().split('\n')[:-1]
data5_2 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data5_2])
classes5_2 = np.array([int(line.split()[2]) for line in row_data5_2])

tree5_2 = split(data5_2, classes5_2)

#The tree of D2.txt is:
decision_boundries = []
format_tree(tree5_2)

db2 = copy.deepcopy(decision_boundries)

#6
#D1.txt:
plt.figure(figsize=(8, 6))
plt.scatter(data5_1[classes5_1 == 0][:, 0], data5_1[classes5_1 == 0][:, 1], s = 20)
plt.scatter(data5_1[classes5_1 == 1][:, 0], data5_1[classes5_1 == 1][:, 1], s = 20)
plt.legend(['label=0', 'label=1'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
for db in db1:
    if db[0] == 'x1':
        plt.axvline(float(db[1]), color = 'red')
    else:
        plt.axhline(float(db[1]), color = 'red')
plt.show()

#D2.txt:
data5_2
plt.figure(figsize=(8, 6))
plt.scatter(data5_2[classes5_2 == 0][:, 0], data5_2[classes5_2 == 0][:, 1], s = 20)
plt.scatter(data5_2[classes5_2 == 1][:, 0], data5_2[classes5_2 == 1][:, 1], s = 20)
plt.legend(['label=0', 'label=1'])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
for db in db2:
    if db[0] == 'x1':
        plt.axvline(float(db[1]), color = 'red')
    else:
        plt.axhline(float(db[1]), color = 'red')
plt.show()
#7
import random
random.seed(1)
row_data7 = open('Dbig.txt').read().split('\n')[:-1]
data7 = np.array([[float(line.split()[0]), float(line.split()[1])] for line in row_data7])
classes7 = np.array([int(line.split()[2]) for line in row_data7])
index = list(range(len(data7)))
random.shuffle(index)
data7 = data7[index]
classes7 = classes7[index]

ns = [32, 128, 512, 2048, 8192]

#def predict(tree, point):
#    x1, x2 = point

def get_rules(tree, rules = []):
    if type(tree) == str:
        return [tree]
    else:
        return rules + get_rules(tree[0]) + get_rules(tree[1])

rules = get_rules(tree5_2)

def check_rule(rule, point):
    x1, x2 = point
    checks = rule.split(':')[0].split(',')[1:]
    result = True
    for check in checks:
        #print(check)
        if 'x1' in check:
            number = x1
        else:
            number = x2
        if '<' in check:
            if number >= float(check.split('<')[1]):
                result = False
                break
        else:
            if number < float(check.split('>=')[1]):
                result = False
                break
    return result, int(rule[-1])



def predict(tree, data):
    rules = get_rules(tree)
    
    result = [[check_rule(rule, tuple(point)) for rule in rules]for point in data]
    result = [list(filter(lambda x:x[0]==True, item))[0][1] for item in result]
    return np.array(result)

result7 = []
dbs = []
tree7s = []
for s in ns:
    train_data = data7[:s, :]
    train_classes = classes7[:s]
    #print(train_data.shape, train_classes.shape)
    test_data = data7[s:, :]
    test_classes = classes7[s:]
    #print(train_data.shape, train_classes.shape)
    tree7 = split(train_data, train_classes)
    tree7s.append(tree7)
    rules = get_rules(tree7)
    #print(rules)
    decision_boundries = []
    format_tree(tree7)
    db = copy.deepcopy(decision_boundries)
    dbs.append(db)
    n = len(decision_boundries) + len(rules)
    prediction = predict(tree7, test_data)
    error = 1 - sum(test_classes == prediction) / len(test_data)
    result7.append([n, error])

#(1)
df7 = pd.DataFrame(result7)
df7.columns = ['n', 'error']
df7
#(2)
df7.plot(x='n', y='error', xlabel = '$n$', ylabel = '$err_n$',  
           title = 'error vs the number of nodes in the tree')
#(3)
for i in range(len(dbs)):
    db3 = dbs[i]
    plt.figure(figsize=(6, 4))
    plt.scatter(data7[classes7 == 0][:, 0], data7[classes7 == 0][:, 1], s = 20)
    plt.scatter(data7[classes7 == 1][:, 0], data7[classes7 == 1][:, 1], s = 20)
    plt.legend(['label=0', 'label=1'])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    for db in db3:
        if db[0] == 'x1':
            plt.axvline(float(db[1]), color = 'red')
        else:
            plt.axhline(float(db[1]), color = 'red')
    plt.title('decision boundries when training sample size is {}'.format(ns[i]))
    plt.show()


#3 sklearn
from sklearn.tree import DecisionTreeClassifier

result_s = []
for s in ns:
    train_data = data7[:s, :]
    train_classes = classes7[:s]
    #print(train_data.shape, train_classes.shape)
    test_data = data7[s:, :]
    test_classes = classes7[s:]
    #print(train_data.shape, train_classes.shape)
    tree8 = DecisionTreeClassifier(random_state = 1)
    tree8.fit(train_data, train_classes)
    prediction = tree8.predict(test_data)
    n = tree8.tree_.node_count 
    error = 1 - sum(test_classes == prediction) / len(test_data)
    
    result_s.append([n, error])

result_s

#(1)
df_s = pd.DataFrame(result_s)
df_s.columns = ['n', 'error']
df_s

#(2)
df_s.plot(x='n', y='error', xlabel = '$n$', ylabel = '$err_n$',  
           title = 'error vs the number of nodes in the tree')


#4 Lagrange Interpolation
from random import uniform
random.seed(7)
a, b = 0, 1
x_train = np.array([uniform(a,b) for _ in range(100)])
y_train = np.sin(x_train)
x_test = np.array([uniform(a,b) for _ in range(100)])
y_test = np.sin(x_test)

from scipy.interpolate  import lagrange
from numpy.polynomial.polynomial import Polynomial

f = lagrange(x_train, y_train)
RMSE_train = (sum(((Polynomial(f.coef[::-1])(x_train) - y_train)**2))/100)**0.5 
RMSE_test = (sum(((Polynomial(f.coef[::-1])(x_test) - y_test)**2))/100)**0.5 
print('RMSE of the train set:', RMSE_train)
print('RMSE of the test set:', RMSE_test)

stds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
RMSE_trains = []
RMSE_tests = []
np.random.seed(1)
for std in stds:
    noise = np.random.randn(100) * std
    x_train_noise = x_train + noise
    f = lagrange(x_train_noise, y_train)

    RMSE_train = (sum(((Polynomial(f.coef[::-1])(x_train) - y_train)**2))/100)**0.5 
    RMSE_trains.append(RMSE_train)
    RMSE_test = (sum(((Polynomial(f.coef[::-1])(x_test) - y_test)**2))/100)**0.5 
    RMSE_tests.append(RMSE_test)
    print('RMSE of the train set:', RMSE_train)
    print('RMSE of the test set:', RMSE_test)

plt.figure(figsize = (8, 6))
plt.plot(stds, RMSE_trains)
plt.plot(stds, RMSE_tests)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('std')
plt.ylabel('RMSE')
plt.legend(['train data', 'test data'])
plt.title('RMSE on the train data and test data change with the std of noise')
