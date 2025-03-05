import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from openpyxl import Workbook

# Disable TensorFlow v2
tf.disable_v2_behavior()

# ✅ Load MNIST Dataset
mnist = tfds.load('mnist', split=['train', 'test'], as_supervised=True)
train_data, test_data = mnist

# ✅ Preprocessing Function
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0  # Normalize
    images = tf.reshape(images, [784])  # Flatten
    labels = tf.one_hot(labels, depth=10)  # One-hot encode
    return images, labels

# ✅ Hyperparameters to Tune
batch_size_list = [1,10, 100]
epochs_list = [10, 50,100]

# ✅ Training Function
def train_and_evaluate(batch_size, epochs):
    print(f"\nTraining with batch_size={batch_size}, epochs={epochs}")

    # ✅ Prepare Dataset
    train_dataset = train_data.map(preprocess).batch(batch_size)
    test_dataset = test_data.map(preprocess).batch(batch_size)

    # ✅ Placeholders
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])

    # ✅ Initialize Weights & Biases
    weights = {
        'h1': tf.Variable(tf.random_normal([784, 128])),
        'h2': tf.Variable(tf.random_normal([128, 64])),
        'out': tf.Variable(tf.random_normal([64, 10]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([128])),
        'b2': tf.Variable(tf.random_normal([64])),
        'out': tf.Variable(tf.random_normal([10]))
    }

    # ✅ Neural Network Model
    def neural_network(x):
        layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
        return tf.add(tf.matmul(layer2, weights['out']), biases['out'])

    # ✅ Compute Loss & Optimizer
    logits = neural_network(X)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    # ✅ Compute Accuracy
    predictions = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # ✅ Train Model
    loss_curve, acc_curve, val_acc_curve = [], [], []
    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            avg_loss = 0
            total_batches = 0
            iterator = tf.compat.v1.data.make_one_shot_iterator(train_dataset)
            next_batch = iterator.get_next()

            while True:
                try:
                    batch_x, batch_y = sess.run(next_batch)
                    _, c = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                    avg_loss += c
                    total_batches += 1
                except tf.errors.OutOfRangeError:
                    break  

            avg_loss /= total_batches
            train_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})
            val_acc = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y})

            loss_curve.append(avg_loss)
            acc_curve.append(train_acc)
            val_acc_curve.append(val_acc)

            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        end_time = time.time()
        execution_time = end_time - start_time

        # ✅ Evaluate on Test Data
        test_acc = []
        y_true, y_pred = [], []
        iterator = tf.compat.v1.data.make_one_shot_iterator(test_dataset)
        next_batch = iterator.get_next()

        while True:
            try:
                batch_x, batch_y = sess.run(next_batch)
                acc, preds = sess.run([accuracy, predictions], feed_dict={X: batch_x, Y: batch_y})
                test_acc.append(acc)
                y_pred.extend(np.argmax(preds, axis=1))
                y_true.extend(np.argmax(batch_y, axis=1))
            except tf.errors.OutOfRangeError:
                break

        final_test_acc = np.mean(test_acc)
        print(f"Test Accuracy: {final_test_acc:.4f}")

        # ✅ Save Confusion Matrix as Image
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix (Batch={batch_size}, Epochs={epochs})")
        cm_filename = f"confusion_matrix_batch{batch_size}_epochs{epochs}.png"
        plt.savefig(cm_filename)
        plt.close()

        # ✅ Save Loss & Accuracy Curves
        plt.figure(figsize=(12, 4))

        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(loss_curve, label='Train Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")

        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(acc_curve, label='Train Accuracy')
        plt.plot(val_acc_curve, label='Val Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy Curve")

        curve_filename = f"curves_batch{batch_size}_epochs{epochs}.png"
        plt.savefig(curve_filename)
        plt.close()

    return batch_size, epochs, execution_time, final_test_acc, cm_filename, curve_filename

# ✅ Run Experiments & Store Results
results = []
for batch_size in batch_size_list:
    for epochs in epochs_list:
        res = train_and_evaluate(batch_size, epochs)
        results.append(res)

# ✅ Save Results to Excel
df = pd.DataFrame(results, columns=['Batch Size', 'Epochs', 'Execution Time (s)', 'Test Accuracy', 'Confusion Matrix Image', 'Loss/Accuracy Curves'])
df.to_excel("training_results.xlsx", index=False)

print("\n✅ All Results Saved in training_results.xlsx ✅")