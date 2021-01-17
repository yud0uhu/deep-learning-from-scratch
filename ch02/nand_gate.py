import numpy as np

# NANDゲートを実装
def NAND(x1, x2):

    # パラメータの設定
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) # 重みとバイアスだけがANDと違う!
    b = 0.7

    # 重み付き入力の総和を計算
    tmp = np.sum(x*w) + b

    # 出力の設定
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0,0), (1,0), (0,1), (1,1)]:
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
