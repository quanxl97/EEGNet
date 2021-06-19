# from tensorflow.keras.models import Model


import tensorflow as tf
# from tensorflow.keras.models import Model
# import yaml




if __name__ == '__main__':
    print('1')
    hello = tf.constant('hello,TensorFlow!!!!!!!!!!!')
    sess = tf.Session()
    print(sess.run(hello))
    print('2')