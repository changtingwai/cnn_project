import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder("float",[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder("float",[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],'SAME')

#first layer
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#2nd layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#fc layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#drop out
keep_prob = tf.placeholder("float")
h_fc1_dp = tf.nn.dropout(h_fc1,keep_prob)

#out
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_dp,W_fc2)+b_fc2)

#run
cross_entry = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entry)
correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

init = tf.initialize_all_variables()
sess.run(init)
for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(50)
    if i %100 == 0:
        train_accruacy = accuracy.eval(feed_dict={x:batch_x,y_:batch_y,keep_prob:1.0})
        print "step %d,train accuracy %g" %(i,train_accruacy)
    train_step.run(feed_dict={x:batch_x,y_:batch_y,keep_prob:0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
