import numpy as np
import matplotlib.pylab as plt

# シグモイド関数の実装
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # 式(3.6)

# x軸の値を生成
x = np.arange(-10, 10, 0.1)
print(np.round(x, 1))

# シグモイド関数による活性化
y = sigmoid(x)
print(np.round(y, 3))

# 作図
plt.plot(x, y) # 点の位置
plt.title("Sigmoid Function", fontsize = 20) # タイトル
plt.show() # グラフを表示