import tensorflow as tf


def get_model(architecture):

    return model_architecture[architecture]

def nn(x_dict):
    """ Implementation of a shallow neural network."""

    # Extract Input.
    x = x_dict["images"]
    # First Hidden Layer.
    layer_1 = tf.layers.dense(x, 256)
    # Second Hidden Layer.
    layer_2 = tf.layers.dense(layer_1, 256)
    # Output Layer.
    output_layer = tf.layers.dense(layer_2, 10)
    return output_layer

def dnn(x_dict):
    """ Implementation of a deep neural network."""

    # Extract Input.
    x = x_dict["images"]
    # FC Hidden Layer.
    layer_1 = tf.layers.dense(x,500)
    layer_2 = tf.layers.dense(layer_1,350)
    layer_3 = tf.layers.dense(layer_2,250)
    # Batch Normalization Layer.
    layer_4 = tf.layers.batch_normalization(layer_3)
    # FC Hidden Layer.
    layer_5 = tf.layers.dense(layer_4,150)
    layer_6 = tf.layers.dense(layer_5,100)
    layer_7 = tf.layers.dense(layer_6,50)
    # Output Layer.
    output_layer = tf.layers.dense(layer_7,10)
    return output_layer
    
def cnn(x_dict):
    """ Implementation of a convolutional neural network."""
    
    # Obtain and reshape the data.
    x = x_dict["images"]
    x = tf.reshape(x, (-1,28,28,1))

    # First Layer (conv+maxpool)
    conv1 = tf.layers.conv2d(inputs=x, filters=32,
        kernel_size=[5,5],padding="same",activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2],strides=2)
    # Second Layer (conv+maxpool)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64,
        kernel_size=[5,5], padding="same",activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2],strides=2)
    # Reshape pool2 into two dimensions.
    pool2_flat = tf.reshape(pool2, [-1,7*7*64])
    # FC Layer.
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # Dropout regularization.
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    #Logits layer.
    output_layer = tf.layers.dense(inputs=dropout, units=10)
    return output_layer

def svm(x_dict):
    #TODO: Implementation of svm. Need to refer LeCun's Paper for this.
    return None

model_architecture = {
    "nn" : nn,
    "dnn": dnn,
    "cnn": cnn,
    "svm": svm
}
 
