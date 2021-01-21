import numpy as np
from sigmoid import sigmoid

# 3.4.3 実装のまとめ

# 空のディクショナリを作成
dic_data = {}
print(dic_data)

# 新たなキーに値を代入
dic_data['疲れた'] = "赤い雄牛"
print(dic_data) # パラメータ名(キー)を指定するとその値を取り出すことができる

# キーを指定して値を取り出す
living_dead = dic_data['疲れた']
print(living_dead)

# 恒等関数(出力層の活性化関数)の実装:回帰問題では恒等関数,2クラス分類問題ではソフトマックス関数を使うのが一般的
def identity_function(x):
    return x

# パラメータを複製する関数を定義
def init_network():

    # ディクショナリを初期化(空のディクショナリを作成)
    network = {}

    # 第1層のパラメータ
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # key : 値の構文
    network['b1'] = np.array([0.1, 0.2, 0.3])

    # 第2層のパラメータ
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])

    # 第3層のパラメータ
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# パラメータ(重みとバイアス)を取得
dic_data = init_network()
print(dic_data)

# ニューラルネットワーク(順伝播)を定義
def forward(network, x):

    # ディクショナリから各パラメータを取り出す
    W1, W2, W3 = network['W1'], network['W2'], network['W3'] # 重み
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # バイアス
    
    # 第1層の処理
    a1 = np.dot(x, W1) + b1 # 重み付き和の計算
    z1 = sigmoid(a1) # 活性化
    
    # 第2層の処理
    a2 = np.dot(z1, W2) + b2 # 重み付き和の計算
    z2 = sigmoid(a2) # 活性化
    
    # 第3層の処理
    a3 = np.dot(z2, W3) + b3 # 重み付き和の計算
    y = identity_function(a3) # 活性化
    
    return y

# 入力信号
X = np.array([1.0, 0.5])

# 出力
# 恒等関数による活性化
Y = forward(dic_data, X)
print(Y)