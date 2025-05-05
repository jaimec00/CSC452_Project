"""defines the model"""


import tensorflow as tf
from tensorflow.keras import layers, Model

def setup_model(input_shape=(520, 704, 1), num_layers=3, initial_filters=64):
	"""
	Builds a U-Net model with the specified input shape, number of down/up sampling layers,
	and initial number of filters.

	Args:
		input_shape (tuple): Dimensions of the input image (height, width, channels).
		num_layers (int): Number of downsampling/upsampling layers.
		initial_filters (int): Number of filters for the first convolutional block.

	Returns:
		tf.keras.Model: The constructed U-Net model.
	"""
	inputs = layers.Input(shape=input_shape)

	# Encoder path (downsampling)
	skips = []
	x = inputs
	filters = initial_filters
	for i in range(num_layers):
		x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
		x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
		skips.append(x)
		x = layers.MaxPooling2D((2, 2))(x)
		filters *= 2

	# Bottleneck
	x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
	x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

	# Decoder path (upsampling)
	for i in reversed(range(num_layers)):
		filters //= 2
		x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)  # Upsampling
		x = layers.Concatenate()([x, skips[i]])  # Skip connection
		x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
		x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

	# Output layer
	outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

	model = Model(inputs, outputs, name='U-Net')

	return model