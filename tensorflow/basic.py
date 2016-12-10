#/usr/bin/env python

import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)

sess = tf.Session()
print sess.run(a+b)

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

a = tf.constant(2)
b = tf.constant(3)

add = tf.add(a,b)
mul = tf.mul(a,b)

print sess.run(add)
print sess.run(mul)

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1,matrix2)

print sess.run(product)

hello = tf.constant("Hello TF")
print sess.run(hello)
