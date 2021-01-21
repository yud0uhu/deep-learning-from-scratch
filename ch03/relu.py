import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x) # 式(3.7)

# お試し
# print(relu(7))
# print(relu(-3))

# x軸の値
x = np.arange(-5, 5, 0.1)

# ReLU関数の計算
y = relu(x)
print(y)

# 作図
plt.plot(x, y) # 点の位置
plt.title("ReLU Function", fontsize = 20) # タイトル
plt.show() # グラフを表示