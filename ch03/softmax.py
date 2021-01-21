import numpy as np

def softmax(a):
    c = np.max(a) # 入力信号aの最大の値
    exp_a = np.exp(a - c ) # オーバーフロー対策:任意定数cを引く
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y