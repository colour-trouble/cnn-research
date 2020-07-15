"""
Convolutional neural network for recognizing hand written digits (0 - 9) of 28x28 pixels.


Summary:

- 32 feature maps, with a stride of 1x1
- 2 convolutional layers and 2 max-pooling layers
- 1 dropout layer to account for overfitting
- Evaluted using mean squared error
- Optimized using Stochastic Gradient Descent


Process (in order):

1)  Conv2D: 4x4 receptive field, 32 feature maps, ReLU.
    - Output is 32, 25(=28-4+1)x25 pixel images, each with distinct features.
    - No. parameters = 32 * (4 * 4 + 1) =  544
    - Output dim = (None, 25, 25, 32)

2)  Max-pooling: 2x2 pool size, padding uncertained (ASSUMED TO BE VALID).
    - Seperates the image into 2x2 pixels and taking the pixel with maximum value.
    - Default stride equal to pool size.
    - Since padding == "valid", no padding, output is 32, 12x12 images.
    - If padding == "same", padding present, output is 32, 13x13 images.
    - Output dim = (None, 12, 12, 32)

3)  Conv2D: 3x3 receptive field, 32 feature maps, ReLU.
    - Output is 32, 10(=12-3+1)x10 pixel images.
    - No. parameters = 32 * (3 * 3 + 1) =  320
    - Output dim = (None, 10, 10, 32)
    
4)  Max-pooling: 2x2 pool size, padding uncertained (ASSUMED TO BE VALID).
    - Seperates the image into 2x2 pixels and taking the pixel with maximum value.
    - Default stride equal to pool size.
    - Output is 32, 5x5 images.
    - Output dim = (None, 5, 5, 32)

5)  Flatten (None, 5, 5, 32) into (None, 800), a 800-dimensional vector

6)  Dense layer: 10 neurons, Softmax.
    - No. parameters = 800 * 10 = 8000


Accuracy:
94.61% after 20 epochs, batch size of 32 out of 60000 samples.

"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation ,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

(train_samples, train_labels), (test_samples, test_labels) = mnist.load_data()

# Last dimension (1) corresponds to single channel image for greyscale
train_samples = train_samples.reshape(train_samples.shape[0], 28, 28, 1) 
test_samples = test_samples.reshape(test_samples.shape[0], 28, 28, 1)
train_samples = train_samples.astype('float32')
test_samples = test_samples.astype('float32')
# Divide by 255 to normalize entries from a range of 0 to 255
train_samples = train_samples/255
test_samples = test_samples/255

# Apply one hot encoding to class labels (digits 1 to 9)
c_train_labels = np_utils.to_categorical(train_labels, 10)
c_test_labels = np_utils.to_categorical(test_labels, 10)

# Building the network
convnet = Sequential()
# 32 feature maps (equal to the no. of neurons)
convnet.add(Conv2D(32, (4, 4), activation='relu', input_shape=(28,28,1)))
print("Layer 1:", convnet.output_shape)
convnet.add(MaxPooling2D(pool_size=(2,2)))
print("Layer 2:", convnet.output_shape)
# Keras automatically adjusts input shape to match the output shape of previous layer
convnet.add(Conv2D(32, (3, 3), activation='relu'))
print("Layer 3:", convnet.output_shape)
convnet.add(MaxPooling2D(pool_size=(2,2)))
print("Layer 4:", convnet.output_shape)
convnet.add(Dropout(0.3))
convnet.add(Flatten())
print("Layer 5:", convnet.output_shape)
convnet.add(Dense(10, activation='softmax'))

# Compile network
convnet.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])

# Train
convnet.fit(train_samples, c_train_labels, batch_size=32, epochs=20, verbose=1)

metrics = convnet.evaluate(test_samples, c_test_labels, verbose=1)
print("\n%s: %.2f%%" % (convnet.metrics_names[1], metrics[1]*100))

# Predict using test_samples as an example
predictions = convnet.predict(test_samples)
