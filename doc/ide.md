# 開発環境構築
- Keras + Tensorflow, Python

## 1. 仮想マシンインストール
- [anaconda](https://www.anaconda.com/distribution/)
- 特に理由がなければ、Python3系版をインストール
- 任意の仮装環境を作成
    - GUI or via Terminal
```
# conda create -n keras python=3.7
```

## 2. [Tensorflow](https://www.tensorflow.org/install) install
```
pip install tensorflow
```
上記のインストールは2.1.0バージョンnがインストールされる。
windowsの場合、2.1.0では上手く動かないので、2.0.0をインストールすべき

```
pip install tensorflow=2.0.0
```

- Tensorflow正常動作確認
```
import tensorflow as tf

hello = tf.constant('Hello, Tensorflow!')
print(hello)
```


## 3. [Keras](https://keras.io/#installation) install
```
pip install keras
```

- Keras with linear regression
```
import numpy as np 
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers

# 1. create pseudo data y = 2*x0 + 3*x1 + 4
X = np.random.rand(100, 2)
y =  2* X[:,0] + 3 * X[:,1] + 4 + .2*np.random.randn(100) # noise added

# 2. Build model 
model = Sequential([Dense(1, input_shape = (2,), activation='linear')])

# 3. gradient descent optimizer and loss function 
sgd = optimizers.SGD(lr=0.1)
model.compile(loss='mse', optimizer=sgd)

# 4. Train the model 
model.fit(X, y, epochs=100, batch_size=2)
```
