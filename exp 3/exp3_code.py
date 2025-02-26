import tensorflow as tf
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()  # Disable eager execution to use TensorFlow's graph execution

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.reshape(-1, 784) / 255.0, x_test.reshape(-1, 784) / 255.0

# Convert labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Define model hyperparameters
input_size = 784
hidden1_size = 128
hidden2_size = 64
output_size = 10
learning_rate = 0.01
batch_size = 100
epochs = 20

# Define placeholders for input and output
X = tf.compat.v1.placeholder(tf.float32, [None, input_size])
y = tf.compat.v1.placeholder(tf.float32, [None, output_size])

# Initialize weights and biases
weights = {
    'w1': tf.Variable(tf.random.truncated_normal([input_size, hidden1_size], stddev=0.1)),
    'w2': tf.Variable(tf.random.truncated_normal([hidden1_size, hidden2_size], stddev=0.1)),
    'w3': tf.Variable(tf.random.truncated_normal([hidden2_size, output_size], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.zeros([hidden1_size])),
    'b2': tf.Variable(tf.zeros([hidden2_size])),
    'b3': tf.Variable(tf.zeros([output_size]))
}

# Define feed-forward neural network
def neural_network(X):
    layer1 = tf.nn.sigmoid(tf.matmul(X, weights['w1']) + biases['b1'])
    layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['w2']) + biases['b2'])
    output_layer = tf.matmul(layer2, weights['w3']) + biases['b3']
    return output_layer

# Compute logits
logits = neural_network(X)

# Define loss function (cross-entropy)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Define optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Define accuracy metric
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Run session
tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Training loop
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x, batch_y = x_train[i:i+batch_size], y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, y: batch_y})
        
        # Calculate and display loss and accuracy at each epoch
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={X: x_train, y: y_train})
        test_acc = sess.run(accuracy, feed_dict={X: x_test, y: y_test})
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}, Test Accuracy: {test_acc*100:.2f}")
    
    # Compute final train and test accuracy
    final_train_acc = sess.run(accuracy, feed_dict={X: x_train, y: y_train})
    final_test_acc = sess.run(accuracy, feed_dict={X: x_test, y: y_test})
    print(f"Final Train Accuracy: {final_train_acc*100:.2f}")
    print(f"Final Test Accuracy: {final_test_acc*100:.2f}")
    
    print("Training Complete!")