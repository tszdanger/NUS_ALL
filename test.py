from PIL import Image, ImageFilter,ImageEnhance
import tensorflow as tf
import matplotlib.pyplot as plt
import time

def imageprepare():
    file_name='C:\\Users\\skywf\\Desktop\\image\\6.png'
    im = Image.open(file_name)
    #
    # enh_im = ImageEnhance.Brightness(im)
    # brightness = 1.5
    # image_bright = enh_im.enhance(brightness)
    # image_bright.show()
    # im.save('kuaile.gif')
    #
    # enh_col_im = ImageEnhance.Contrast(im)
    # contrast = 1.5
    # image_contrast = enh_im.enhance(contrast)
    # image_contrast.show()
    # im.save('kuaile_1.gif')
    #
    # enh_sha_im = ImageEnhance.Sharpness(im)
    # sharpness = 3
    # image_sharped = enh_im.enhance(sharpness)
    # image_sharped.show()
    # im.save('kuaile_2.png')
    # im.filter(ImageFilter.SHARPEN).save('C:\\Users\\skywf\\Desktop\\image\\kuaile_2_new.png')
    # im.filter(ImageFilter.EDGE_ENHANCE).save('C:\\Users\\skywf\\Desktop\\image\\kuaile_2_new1.png')
    print(im.size)
    print(im.mode)
    im = im.resize((28,28),Image.ANTIALIAS)

    # im = im.filter(ImageFilter.EDGE_ENHANCE)
    im = im.filter(ImageFilter.SHARPEN)
    print(im.size)
    print(im.mode)
    plt.imshow(im)
    plt.show()
    im = im.convert('L')
    im.save("C:\\Users\\skywf\\Desktop\\image\\1.png")

    tv = list(im.getdata())
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    print(tva)
    return tva

result=imageprepare()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "C:\\Users\\skywf\\Desktop\\model.ckpt")

    prediction=tf.argmax(y_conv,1)
    predint=prediction.eval(feed_dict={x: [result],keep_prob: 1.0}, session=sess)
    print(h_conv2)

    print('识别结果:')
    print(predint[0])

