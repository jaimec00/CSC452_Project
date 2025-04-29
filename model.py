"""defines the model"""


from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD


def setup_model():

    # create model
    model = Sequential()

    # Our First Convolution Layer, Filter size 32 which reduces our layer size to 26 x 26 x 32
    # We use ReLU activation and specify our input_shape which is 28 x 28 x 1
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    # Our Second Convolution Layer, Filter size 64 which reduces our layer size to 24 x 24 x 64
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # We use MaxPooling with a kernel size of 2 x 2, this reduces our size to 12 x 12 x 64
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # We then Flatten our tensor object before input into our Dense Layer
    # A flatten operation on a tensor reshapes the tensor to have the shape that is
    # equal to the number of elements contained in tensor
    # In our CNN it goes from 12 * 12 * 64 to 9216 * 1
    model.add(Flatten())

    # We connect this layer to a Fully Connected/Dense layer of size 1 * 128
    model.add(Dense(128, activation='relu'))

    # We create our final Fully Connected/Dense layer with an output for each class (10)
    model.add(Dense(num_classes, activation='softmax'))

    return model