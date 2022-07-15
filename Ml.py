import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# creating features
x = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])
# creating lables
y= np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])
# visualize it
plt.scatter(x,y)
# Turning numpy array in tensor
X=tf.constant(x,dtype=tf.float32)
Y=tf.constant(y,dtype=tf.float32)
X,Y
# creating a model
model=tf.keras.Sequential([
                  tf.keras.layers.Dense(100, activation="relu"),   
                  tf.keras.layers.Dense(100, activation="relu"),                         
                  tf.keras.layers.Dense(100, activation="relu"),                         
                  tf.keras.layers.Dense(1)
])
# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=["mae"])
# fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=1000)
model.predict([17.0])