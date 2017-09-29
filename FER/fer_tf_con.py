import tensorflow as tf
import csv
import numpy as np

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

#Credits flatten_layer fucntion : https://github.com/sdhayalk
def flatten_layer(layer):
    # flatten tensor of 4 dimension to 2 dimension so that they can be used for fully connected layer
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    #layer_shape == [num_images, img_height, img_width, num_channels]
    # The number of features is:
    #img_height * img_width * num_channels, we can use a function from TensorFlow to calculate this
    num_features = layer_shape[1:4].num_elements()

    layer_flat = tf.reshape(layer, [-1, num_features])  # Reshape the layer to [num_images, num_features].

    return layer_flat, num_features

def load_dataset(file):
    dataset_features = []
    dataset_labels = []

    #file = '/input' + file

    with open(file) as csvfile:
        csv_reader_object = csv.reader(csvfile, delimiter=',')
        count = 0
        next(csv_reader_object, None)
        for row in csv_reader_object:
            if len(row) == 0 :
                _0 = 0  # ignore
            else:
                #print(row)
                dataset_features.append(row[1].split())
                # print(count)
                # count += 1
                temp = np.zeros(7, dtype=int)
                temp[int(row[0])] = int(1)
                dataset_labels.append(temp)

    return np.array(dataset_features), np.array(dataset_labels)

def get_next_batch(dataset_features, dataset_labels, batch_index, batch_size):
    return dataset_features[batch_index*batch_size:(batch_index+1)*batch_size, :], dataset_labels[batch_index*batch_size : (batch_index+1)*batch_size, :]

def conv_net(x, weights, biases, dropout):

    x = tf.reshape(x, [-1, 48, 48, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print(conv1.get_shape())
    conv1 = maxpool2d(conv1, k=2)
    print(conv1.get_shape())

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print(conv2.get_shape())
    conv2 = maxpool2d(conv2, k=2)
    print(conv2.get_shape())

    '''conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    print(conv3.get_shape())
    conv3 = maxpool2d(conv3, k=2)
    print(conv3.get_shape())

    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'])
    print(conv4.get_shape())
    conv4 = maxpool2d(conv4, k=2)
    print(conv4.get_shape())'''

    #flatten_layer
    layer_flat, num_features = flatten_layer(conv2)

    # Fully connected layer
    weights['wd1'] = tf.Variable(tf.random_normal([num_features, 1024]))

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print(out.get_shape())
    return out

# Parameters
learning_rate = 0.001
epochs = 1
batch_size = 50

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
#test_valid_size = 256

# Network Parameters
n_classes = 7  # FER total classes
dropout = 0.75  # Dropout, probability to keep units

train_features,train_labels = load_dataset("Data/training.csv")
train_features = train_features.astype(int)
train_features = train_features/255.0
test_features,test_labels = load_dataset("Data/test.csv")
test_features = test_features.astype(int)
test_features = test_features/255.0
validation_features,validation_labels = load_dataset("Data/testprivate.csv")
validation_features = validation_features.astype(int)
validation_features = validation_features/255.0

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wd1': tf.Variable(tf.random_normal([3*3*512, 1024])),
    'out': tf.Variable(tf.random_normal([1024, n_classes]))}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bc3': tf.Variable(tf.random_normal([256])),
    'bc4': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))}

# tf Graph input
x = tf.placeholder(tf.float32, [None, 48*48])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)#default learning_rate = 0.001

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf. global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        total_no_of_batches = int(train_features.shape[0] / batch_size)
        for batch in range(total_no_of_batches):
            batch_x, batch_y = get_next_batch(train_features,train_labels,batch, batch_size)
            print(batch_x.shape,batch_y.shape)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: validation_features,
                y: validation_labels,
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} -'
                  'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: test_features,
        y: test_labels,
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))
