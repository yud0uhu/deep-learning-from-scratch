import sys, os
sys.path.append(os.pardir) # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist

# 最初の呼び出しは数分待ちます...
(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

# それぞれのデータの形状を出力
print(x_train.shape) # (6000, 784) # 訓練画像
print(t_train.shape) # (6000,) # 訓練ラベル
print(x_test.shape) # (10000, 784) # テスト画像
print(t_test.shape) # (10000,) # テストラベル
