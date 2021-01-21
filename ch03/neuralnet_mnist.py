import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

from sigmoid import sigmoid
from softmax import softmax

#  MNISTデータを読み込む関数を実装
def get_data():
    
    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)
    
    return x_test, t_test

# 学習済みのパラメータを読み込む関数を実装
def init_network():
    
    # ライブラリを読み込む
    import pickle
    
    # ファイルの位置を指定
    file_path = "sample_weight.pkl"
    
    # 学習済みのパラメータの読み込み
    with open(file_path, 'rb') as f:
        network = pickle.load(f)
        
        return network

# 手書き数字から正解を予測する関数を実装
def predict(network, x):
    
    # 学習済みパラメータを抽出
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 第1層の処理
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    
    # 第2層の処理
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    
    # 第3層の処理
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

# テスト画像とテストラベルを読み込む
x, t = get_data()

# 学習済みパラメータを読み込む
network = init_network()

# データ数を確認
print(len(x)) # 要素数を取得
print(len(t)) # 要素数を取得
print(network.keys()) # ディクショナリ型データのキーを取得

# 推論
y = predict(network, x[0])
print(y)
print(np.round(y, 3)) # (分かりやすいように値を丸めた版)
print(np.sum(y)) # (一応総和が1となることを確認)

# 最大値を取り出す
p = np.argmax(y)
print("期待する値(文字):") # 確率が最も大きいもの
print(p)

# 1つ目の正解データ
print("ニューラルネットワークが予測した値:")
print(t[0])

# 予測結果と正解ラベルを比較
p == t[0]

# 正解数のカウントを初期化
accuracy_cnt = 0

# データ数回推定を行う
for i in range(len(x)):
    
    # 予測
    y = predict(network, x[i])
    
    # 確率が最大のものを取り出す
    p = np.argmax(y)
    
    # 正解数をカウント
    if p == t[i]:
        accuracy_cnt += 1

# 正解率を計算
accuracy_rate = accuracy_cnt / len(x) # (正解数) / (データ数)
print(accuracy_rate)

# (折角なので人間にとって分かりやすく表示しましょうか)
print("Accurary:" + str(np.round(accuracy_rate * 100, 2)) + "%")