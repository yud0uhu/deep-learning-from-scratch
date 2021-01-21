import numpy as np
import matplotlib.pylab as plt
from sigmoid import sigmoid
from step_function import step_function
from relu import relu

# 共通のx(軸)の値
x = np.arange(-3, 3, 0.1)

# 活性化
y_step = step_function(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)

# 作図
plt.plot(x, y_step, linestyle = "--", label = "Step") # ステップ関数のグラフ
plt.plot(x, y_sigmoid, label = "Sigmooid") # シグモイド関数のグラフ
plt.plot(x, y_relu, linestyle = ":", label = "ReLU") #ReLU関数のグラフ
plt.legend() # 凡例
plt.show() # グラフを表示