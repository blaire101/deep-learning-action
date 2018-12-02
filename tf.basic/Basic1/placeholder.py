import tensorflow as tf

# 在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32, [None], name='x1')
input2 = tf.placeholder(tf.float32, [], name='x2')

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output
# ouput = tf.multiply(input1, input2)
output = input1

batch_size = tf.constant(5)

start_tokens = tf.ones([batch_size, ], tf.int32)


with tf.Session() as sess:
    print(sess.run(tf.shape(input1), feed_dict={input1: [8., 2.0]}))

    # print(sess.run(tf.shape(input1)))

    print(sess.run(input2, feed_dict={input2: 9.}))

    print(sess.run(tf.shape(input2), feed_dict={input2: 9.}))

    print(sess.run(start_tokens))
