from __future__ import absolute_import, division, print_function, unicode_literals
try:
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# random seed for always same results
tf.random.set_seed(678)

import numpy as np
print(tf.__version__)

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = np.array([0.,1.,1.,0.])
#Two dense Layers
model = Sequential()
# first dense layer
model.add(Dense(units=2,activation='sigmoid',input_dim=2))
# second dense layer
model.add(Dense(units=1,activation='sigmoid'))
# loss function and optimization
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
# model summary
print(model.summary())

for layer in model.layers:
    print(layer.output)  # does not fail

# Train (takes about 3 minutes from Colab)
model.fit(X,y,epochs=50000,batch_size=4,verbose=0)

print(model.predict(X,batch_size=4))


print("first layer weights: ",model.layers[0].get_weights()[0])
print("first layer bias: ",model.layers[0].get_weights()[1])







#### calculate yourself and verify same result with TF dense layers

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def get_output(x):
    layer0 = model.layers[0]
    # first dense layer, first node output
    layer0_weights, layer0_bias = layer0.get_weights()
    layer0_node0_weights = np.transpose(layer0_weights)[0]
    layer0_node0_bias = layer0_bias[0]
    layer0_node0_output = sigmoid( np.dot( x, layer0_node0_weights ) + layer0_node0_bias )
    # second dense layer, second node output
    layer0_node1_weights = np.transpose(layer0_weights)[1]
    layer0_node1_bias = layer0_bias[1]
    layer0_node1_output = sigmoid( np.dot( x, layer0_node1_weights ) + layer0_node1_bias )
    # second layer output
    layer1 = model.layers[1]
    layer1_weights, layer1_bias = layer1.get_weights()
    layer1_output = sigmoid( np.dot( [layer0_node0_output, layer0_node1_output], layer1_weights ) + layer1_bias )

    print(layer1_output)

get_output([0,0])
get_output([0,1])
get_output([1,0])
get_output([1,1])
