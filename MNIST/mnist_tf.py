import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from helper import batches


learning_rate = 0.1
n_input = 784
n_classes = 10
n_hidden = 256
no_epochs = 100
batch_size = 128

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.reset_default_graph()

print("Data Downloaded")

train_features = mnist.train.images
train_labels = mnist.train.labels

test_features = mnist.test.images
test_labels = mnist.test.labels

print("Data loaded")

sess = tf.InteractiveSession()

#placeholder is like a node created , in which the value will be entered during tensorflow computation
features = tf.placeholder(tf.float32, shape=[None, n_input])#None here will be number of examples
labels = tf.placeholder(tf.float32, shape=[None, n_classes])#None here will be number of examples

#truncated normal is used to have mean as Zero and values between 2 S.D
W_hidden = tf.Variable(tf.truncated_normal([n_input,n_hidden]))
b_hidden = tf.Variable(tf.truncated_normal([n_hidden]))

#use names of W and b because they will be saved into the model and then can be referenced using those names
W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name = "weight_0")
b = tf.Variable(tf.truncated_normal([n_classes]),name = "bias_0")

sess.run(tf.global_variables_initializer())

#create the model y = x*W + b
hidden_layer = tf.add(tf.matmul(features,W_hidden),b_hidden)
hidden_layer = tf.nn.relu(hidden_layer)

logits = tf.add(tf.matmul(hidden_layer,W),b)

#use cross-entropy to get the loss.
#cross-entropy is given by multiplying the softmax probablities by one-hot encoded labels vector

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

#Note that tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized
#model prediction and sums across all classes, and tf.reduce_mean takes the average over these sums.

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf	.argmax(labels, 1))

#correct_prediction now has the list of boolean values
#we cast them to integers and then take the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epochs in range(no_epochs):
	total_no_of_batches = int(mnist.train.num_examples/batch_size)
	for i in range(total_no_of_batches):
		batch = mnist.train.next_batch(batch_size)
		optimizer.run(feed_dict={features: batch[0], labels: batch[1]})#optimizer is defined in main
	if(epochs % 10 == 0):
		cost = sess.run(cross_entropy,feed_dict = {features: batch[0],labels : batch[1]})
		print("Loss : ",epochs," - ",cost)
		accuracy_epoch = sess.run(accuracy,feed_dict={features : mnist.validation.images,labels: mnist.validation.labels})
		print("Accuracy : ",epochs," - ",accuracy_epoch)

#now lets save the model
save_file = './train_model.ckpt'

saver = tf.train.Saver()

saver.save(sess, save_file)
print('Trained Model Saved.')

# for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
#     sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

print(accuracy.eval(feed_dict={features: test_features, labels: test_labels}))
