from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

inputs = Input(shape=(1,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs, output)

sgd = optimizers.SGD(lr=0.01) # 하이퍼 파라미터를 조작할 수 있음.

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
# linear_model.fit(X, y, epochs=300)

test_X = [1.5,2.3,7.1,8.6]
X = [1, 2, 3, 4, 5, 6, 7, 8, 9] # 공부하는 시간
y = [11, 22, 33, 44, 53, 66, 77, 87, 95] # 각 공부하는 시간에 맵핑되는 성적


linear_model.fit(X, y, epochs=1)

print(linear_model.predict(test_X))