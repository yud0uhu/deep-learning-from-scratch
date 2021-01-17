import numpy as np
from and_gate import AND
from nand_gate import NAND
from or_gate import OR

# XORゲートを実装
def XOR(x1, x2):

    # 第1層
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)

    # 第2層
    y = AND(s1, s2)

    # 出力
    return y

if __name__=='__main__':
    for xs in [(0,0), (1,0), (0,1), (1,1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
