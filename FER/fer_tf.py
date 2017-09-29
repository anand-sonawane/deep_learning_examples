import tensorflow as tf
import numpy as np
import csv

learning_rate = 0.01
#n_input = 48
n_input = 2304
n_classes = 7
n_hidden = 512
no_epochs = 20
batch_size = 20

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


train_features,train_labels = load_dataset("Data/training.csv")
test_features,test_labels = load_dataset("Data/test.csv")
validation_features,validation_labels = load_dataset("Data/testprivate.csv")

print("Data loaded")

sess = tf.InteractiveSession()

#create placeholders
features = tf.placeholder(tf.float32,shape = [None,n_input])
labels = tf.placeholder(tf.float32,shape = [None,n_classes])

#hidden weight and biases

W_hidden = tf.Variable(tf.truncated_normal([n_input,n_hidden]))#input and output length is passed to truncated_normal
b_hidden = tf.Variable(tf.truncated_normal([n_hidden]))

#weight and biases

W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name = "weight_fer")
b = tf.Variable(tf.truncated_normal([n_classes]),name = "bias_fer")

sess.run(tf.global_variables_initializer())

#create the model y = x*W + b
hidden_layer = tf.add(tf.matmul(features,W_hidden),b_hidden)
hidden_layer = tf.nn.relu(hidden_layer)

logits = tf.add(tf.matmul(hidden_layer,W),b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf	.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epochs in range(no_epochs):
	total_no_of_batches = int(train_features.shape[0] / batch_size)
	for i in range(total_no_of_batches):
		batch_x, batch_y = get_next_batch(train_features,train_labels,i, batch_size)
		optimizer.run(feed_dict={features: batch_x, labels: batch_y})#optimizer is defined in main
	if(epochs % 1 == 0):
		cost = sess.run(cross_entropy,feed_dict = {features: batch_x,labels : batch_y})
		print("Loss : ",epochs," - ",cost)
		accuracy_epoch = sess.run(accuracy,feed_dict={features : validation_features,labels: validation_labels})
		print("Accuracy : ",epochs," - ",accuracy_epoch)

#now lets save the model
save_file = './train_model_fer.ckpt'

saver = tf.train.Saver()

saver.save(sess, save_file)
print('Trained Model Saved.')

# for batch_features, batch_labels in batches(batch_size, train_features, train_labels):
#     sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

print(accuracy.eval(feed_dict={features: test_features, labels: test_labels}))
