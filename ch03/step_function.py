import numpy as np
import matplotlib.pylab as plt

# ステップ関数の実装
def step_function(x):
    return np.array(x > 0, dtype = np.int)
    
#    # 式(3.3)
#    if x > 0:
#        # 引数xが0より大きい値の場合
#        return 1
#    else:
#        # 引数xが0以下の値の場合
#        return 0

# x軸の値を生成
# -5から5までのx軸をプロット
x = np.arange(-5, 5, 0.1)
# print(np.round(x, 1))

# ステップ関数による活性化
y = step_function(x)
print(y)

# 作図
plt.plot(x, y) # 点の位置
plt.title("step Function", fontsize = 20) # タイトル
plt.show() # グラフを表示