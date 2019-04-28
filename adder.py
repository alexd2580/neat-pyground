import tensorflow as tf
import random


def random_int():
    return int(random.random() * 10)


def generate_example():
    a = random_int()
    b = random_int()
    result = a + b
    # const_in = tf.constant([[
    #     1 if i == a or i == b + 10 else 0
    #     for i in range(20)
    # ]])
    # const_out = tf.constant([[
    #     1 if i == result else 0
    #     for i in range(20)
    # ]])
    const_in = [
        1 if i == a or i == b + 10 else 0
        for i in range(20)
    ]
    const_out = [
        1 if i == result else 0
        for i in range(20)
    ]
    return a, b, result, const_in, const_out


def generate_feed_dict(input_var, output_var):
    data = [generate_example() for _ in range(BATCH_SIZE)]
    batch_in = [in_tensor for _, _, _, in_tensor, _ in data]
    batch_out = [out_tensor for _, _, _, _, out_tensor in data]
    return {input_var: batch_in, output_var: batch_out}


# Create the network.

# Terminal layers.
input = tf.placeholder(tf.float32, [None, 20])
expected_output = tf.placeholder(tf.float32, [None, 20])

# First layer.
weights1 = tf.Variable(
    tf.random_normal([20, 20], stddev=0.03),
    name='weights1',
)
bias1 = tf.Variable(tf.random_normal([20]), name='bias1')
hidden1 = tf.nn.sigmoid(tf.add(bias1, tf.matmul(input, weights1)))

# Second layer.
weights2 = tf.Variable(
    tf.random_normal([20, 20], stddev=0.03),
    name='weights2',
)
bias2 = tf.Variable(tf.random_normal([20]), name='bias2')
hidden2 = tf.nn.sigmoid(tf.add(bias2, tf.matmul(hidden1, weights2)))

# Second layer.
weights3 = tf.Variable(
    tf.random_normal([20, 20], stddev=0.03),
    name='weights3',
)
bias3 = tf.Variable(tf.random_normal([20]), name='bias3')
output = tf.nn.sigmoid(tf.add(bias3, tf.matmul(hidden2, weights3)))

# TODO Understand cost function.
output_clipped = tf.clip_by_value(output, 1e-10, 0.9999999)
cost = -tf.reduce_mean(
    tf.reduce_sum(
        expected_output * tf.log(output_clipped) +
        (1 - expected_output) * tf.log(1 - output_clipped),
        axis=1
    ),
)

# Add the optimizer.
LEARNING_RATE = 0.5
optimiser = tf.train \
    .GradientDescentOptimizer(learning_rate=LEARNING_RATE) \
    .minimize(cost)

init_op = tf.global_variables_initializer()

# Accessory output.
actual_output = tf.math.argmax(output, 1)
correctness_bool_list = tf.equal(actual_output, tf.math.argmax(expected_output, 1))
correctness_list = tf.cast(correctness_bool_list, tf.float32)
correctness = tf.reduce_mean(correctness_list)

# Add the correctness to the summary.
tf.summary.scalar('correctness', correctness)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary')

with tf.Session() as sess:
    sess.run(init_op)

    LEARNING_SET_SIZE = 100000
    BATCH_SIZE = 50
    num_batches = int(LEARNING_SET_SIZE / BATCH_SIZE)

    EPOCHS = 10
    for epoch in range(EPOCHS):
        avg_cost = 0
        for i in range(num_batches):
            feed_dict = generate_feed_dict(input, expected_output)
            _, c = sess.run([optimiser, cost], feed_dict=feed_dict)
            avg_cost += c / num_batches

        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
        feed_dict = generate_feed_dict(input, expected_output)
        summary = sess.run(merged, feed_dict=feed_dict)
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    feed_dict = generate_feed_dict(input, expected_output)
    print(sess.run(correctness, feed_dict=feed_dict))

    for i in range(100):
        a, b, sum_ab, in_vals, _ = generate_example()
        feed_dict = {input: [in_vals]}
        nn_out = sess.run(actual_output, feed_dict=feed_dict)
        print(f"{a}\t+ {b}\t= {sum_ab}\t?= {nn_out}")
