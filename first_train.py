
# import numpy as np
# #将文本文件转换为numpy矩阵
# with open("prices.txt","r") as file:
#     data = np.array([line.strip().split(",") for line in file],dtype=np.float32)
#
# length =len(data)
# #求出数据长度
# n_train,n_cv = int(0.7*length),int(0.15*length)
# #选出训练集和交叉集
# idx = np.random.permutation(length)
# #随机产生一个序列
# train_idx,cv_idx = idx[:n_train],idx[n_train:n_train+n_cv]
# test_idx = idx[n_train+n_cv:]
# #把几个集都给选好
# train,test,cv = data[train_idx],data[test_idx],data[cv_idx]
#
# #以上完成了把数据从txt文本读出来并且完成分类的过程
# x_test = test[:,0]
# y_test = test[:,1]
# y_train = train[:,1]
# x_train = train[:,0]
# print(x_train)
# print(y_train)
#
# from sklearn.linear_model import logistic
# clf = logistic.LogisticRegression()
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(y_pred)
#
# a = np.array([1,2,3,4],dtype=float)
# b = np.array([[1,2],[3,4],[5,6]])
# c = np.array([1,2,3,4],ndmin=2)
# d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(d)
# print(d.shape)
# e = d.reshape(2,4)
# print(e)
#
#
#
#import tensorflow as tf
#import numpy as input

#
# a = tf.constant(3.0,dtype=tf.float32)
# b = tf.constant(4.0,dtype=tf.float32)
# total = a+b
# print('a is {}'.format(a))
# print('b is {}'.format(b))
# print('total is {}'.format(total))
# sess = tf.Session()
# result = sess.run({'a':a,'b':b,'total':total})
# print(result)
#
# print('a is {}'.format(a))
# print('b is {}'.format(b))
# print('total is {}'.format(total))
#
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# z = x+y
#
# n = sess.run(z,feed_dict={x:3,y:4.5})
# v = sess.run(z,feed_dict={x:[1,3],y:[2,4]})
#
# print('3+4.5 is {}'.format(n))
# print('[1,3]+[2,4] will be {}'.format(v))
#
# my_data = [[0,1],[2,3],[4,5],[6,7] ]
# slices = tf.data.Dataset.from_tensor_slices(my_data)
# next_item = slices.make_one_shot_iterator().get_next()
#
# while 1:
#     try:
#         print(sess.run(next_item))
#     except tf.errors.OutOfRangeError:
#         break

# sess = tf.Session()
# x = tf.placeholder(tf.float32, shape = [None,3])
# linear_model = tf.layers.Dense(units = 1)
# y = linear_model(x)
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# out = sess.run(y,feed_dict={x:[[1,2,3],[4,5,6]]})
#
# print('output from dense layer: {}'.format(out))

# x = tf.constant([[1],[2],[3],[4]],dtype= tf.float32)
# y_true = tf.constant([[0],[-1],[-2],[-3]],dtype = tf.float32)
# linear_model  = tf.layers.Dense(units = 1)
# y_pred = linear_model(x)
#
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print('predicted value is {}'.format(sess.run(y_pred)))
#
# loss = tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)
# print('loss is {}'.format(sess.run(loss)))
#
# optimizer = tf.train.GradientDescentOptimizer(0.01)
# train = optimizer.minimize(loss)
# for i in range(100):
#     _, loss_value = sess.run((train,loss))
#     print('the final {}result loss is now {}'.format(i,loss_value))
#-----------------------------------------------------------------------------
# class Perceptron:
#     curr_epoch = 0
#     curr_error = 99e99
#
#     #2x-1
#     def step(self,x):
#         is_greater  = tf.greater(x,0)
#         as_float = tf.to_float(is_greater)
#         doubled = tf.multiply(as_float,2)
#         return tf.subtract(doubled,1)
#
#     def id(self,x):
#         return x
#
#     def __init__(self,num_in,xfer=0):
#         self.in_count = num_in
#         #weights
#         self.w = tf.Variable(tf.random_normal([num_in,1]))
#
#         self.IN = tf.placeholder(tf.float32,[None,num_in])
#         self.OUT = tf.placeholder(tf.float32,[None,1])
#
#         if not xfer:
#             self.act = self.step
#         else:
#             self.act = self.id
#
#         self.sess = tf.Session()
#         self.init = tf.global_variables_initializer()
#         self.sess.run(self.init)
#
#
#     def _feedforward(self,x):
#         return self.act(tf.matmul(x,self.w))
#
#     def predict(self,x):
#         temp_a = self.sess.run(self._feedforward(x))
#         print('the predicted is {}'.format(temp_a))
#         return temp_a
#
#
#     def train(self,train_in,train_out,alpha =0.01,max_epochs=10,max_error=0.01,restart=0):
#         output = self._feedforward(self.IN)
#
#         error = tf.subtract(self.OUT,output)
#
#         mse = tf.reduce_mean(tf.square(error))
#
#         delta = tf.matmul(self.IN,error,transpose_a=True)
#
#         ldelta = tf.multiply(alpha,delta)
#
#         train = tf.assign(self.w,tf.add(self.w,ldelta))
#
#         print('-------start---------')
#
#         self.curr_epoch=0
#         self.curr_error=99.9
#
#         if restart:
#             self.sess.run(self.init)
#
#         while self.curr_epoch<max_epochs and self.curr_error>max_error:
#             self.curr_epoch +=1
#             self.curr_error, _ = self.sess.run([mse,train],feed_dict = {self.IN:train_in,self.OUT:train_out})
#
#             print('Epoch{} of{} : and the error is{}'.format(self.curr_epoch,max_epochs,self.curr_error) )
#
#         return self.curr_error
#--------------------------------------------------------------------------------

# from keras.models import Sequential
#
# model = Sequential()
#
# from keras.layers import Dense
#
# model.add(Dense(units=64,activation='relu',input_dim=100))
# model.add(Dense(units=10,activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
#
# x_train = 1
# y_train = 2
# model.fit(x_train,y_train,batch_size=32,epochs=5)
#
#
#
#
#
# import numpy as np
# import sklearn
# from sklearn.datasets import fetch_20newsgroups
#
# twenty_train = fetch_20newsgroups(subset='train')
# print(twenty_train.target_names)
# len(twenty_train.data)
#
# for article in twenty_train.data[:3]:
#     myindex = twenty_train.data.index(article)

    # print('\n article {} lable:{}\n '.format(myindex, twenty_train.target_names[twenty_train.data[myindex]]))
    # print(article)

# ----------------------------------------------
#just NB
#from sklearn.feature_extraction.text import CountVectorizer
# count_vect  =CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# print('X_train_counts is{}'.format(X_train_counts.shape))
# print('X_train_counts[0] is {}'.format(X_train_counts[0]))
#
# A = X_train_counts[0]
# # for index in A.indices[A.indptr[0]:A.indptr[1]]:
# #     print(count_vect.get_feature_names()[index])
#
# print(count_vect.get_feature_names()[A.indices[88]])

#from sklearn.naive_bayes import MultinomialNB
# clf = MultinomialNB().fit(X_train_counts,twenty_train.target)
#
# twenty_test = fetch_20newsgroups(subset='test')
# print(len(twenty_test.data))
#
# X_test_counts = count_vect.transform(twenty_test.data)
#
# predicted = clf.predict(X_test_counts)
# print('ACCURACYH: {}'.format(np.mean(predicted == twenty_test.target)))
#
#
#
# from sklearn.feature_extraction.text import TfidfTransformer
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# tfidf_clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# predicted = tfidf_clf.predict(X_test_tfidf)
# print('accuracy(tfidf) is{}'.format(np.mean(predicted==twenty_test.target)))
#
#
# print('\n')
#
# tfidf_transformer = TfidfTransformer()
# sw_count_vect = CountVectorizer(stop_words='english')
# X_train_counts = sw_count_vect.fit_transform(twenty_train.data)
# X_test_counts = sw_count_vect.transform(twenty_test.data)
#
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
#
# tfidf_clf = MultinomialNB().fit(X_train_tfidf,twenty_train.target)
# predicted = tfidf_clf.predict(X_test_tfidf)
#
# print('accuracy(tfidf-stopwords) is {}'.format(np.mean(predicted == twenty_test.target)))

# X = ['hello world','this is test']
# cv = CountVectorizer()
# X_count = cv.fit_transform(X)
# X_target = [1,0]
# #训练样本先是fit得到一系列参数，然后再做trans
# #这里其实应该用one-hot编码的
# trivial_classifier = MultinomialNB().fit(X_count,X_target)
#
# Y = ['test','hello','world']
# Y_count = cv.transform(Y)
# Y_target = [0,2,2]
# predicted = trivial_classifier.predict(Y_count)
# print('accuracy my own is {}'.format(np.mean(predicted ==Y_target)))
#
#
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
# cv = CountVectorizer()#创建词袋数据结构
# cv_fit=cv.fit_transform(texts)
# #上述代码等价于下面两行
# #cv.fit(texts)
# #cv_fit=cv.transform(texts)
#
# print(cv.get_feature_names())    #['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典
#
# print(cv.vocabulary_	)              # {‘dog’:2,'cat':1,'fish':3,'bird':0} 字典形式呈现，key：词，value:词频
#
# print(cv_fit)
#
# # （0,3） 1   第0个列表元素，**词典中索引为3的元素**， 词频
# #（0,1）1
# #（0,2）1
# #（1,1）2
# #（1,2）1
# #（2,0）1
# #（2,3）1
# #（3,0）1
#
# print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
# #[[0 1 1 1]
# # [0 2 1 0]
# # [1 0 0 1]
# # [1 0 0 0]]
#
# print(cv_fit.toarray().sum(axis=0))  #每个词在所有文档中的词频
# #[2 3 2 2]
#
#



#
#
# # --------------------------------------------
# # SVM
#
# from sklearn.linear_model import SGDClassifier
# svm_clf = SGDClassifier(alpha=1e-3,max_iter=5,random_state=42,penalty='l2')
#
# svm_clf.fit(X_train_tfidf,twenty_train.target)
# predicted = svm_clf.predict(X_test_tfidf)
#
# print('svm accuracy:{}'.format(np.mean(predicted == twenty_test.target)))
#
# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf',TfidfTransformer() ),('clf',MultinomialNB()),])
#
# text_clf.fit(twenty_train.data,twenty_train.target)
# predicted = text_clf.predict(twenty_test.data)
#
# print('accuracy pipeline nb is {}'.format(np.mean(predicted==twenty_test.target)))
#
# text_svm = Pipeline([('vect',CountVectorizer(stop_words='english')),('tfidf',TfidfTransformer()),('svm',SGDClassifier(alpha=1e-3,max_iter=5,random_state=42)),])
# text_svm.fit(twenty_train.data,twenty_train.target)
# predicted = text_svm.predict(twenty_test.data)
# print('accuracy PIP SVM is {}'.format(np.mean(predicted == twenty_test.target)))
#

'''

exercise 1

'''

# ----------------------------


# from __future__ import  print_function
# 这句话在py2记得加上
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.keras.optimizers import SGD
#
#
# import numpy as np
#
# def createModel(input_shape,hid_size,num_outputs):
#     model = Sequential()
#     model.add(Dense(hid_size, activation='relu', input_shape=input_shape))
#     model.add(Dense(num_outputs,activation='sigmoid'))
#     return model
#
# def train(model,x_train,y_train,steps,epochs,x_test,y_test,modelname):
#     model.compile(optimizer = SGD(lr=0.7,momentum=0.3),loss = 'mean_squared_error',metrics = ['accuracy'])
#     save_callback = ModelCheckpoint(filepath=modelname)
#     early_stop = EarlyStopping(monitor = 'loss',min_delta=0.01,patience=20)
#
#     model.fit(x = x_train,y = y_train,steps_per_epoch=steps,epochs=epochs,shuffle=True,callbacks=[save_callback,early_stop])
#     print(model.evaluate(x = x_test,y = y_test))
#
# def predict(model,x):
#     print(model.predict(x))
#
# def main():
#     x = [(0.,0.),(1.,0.),(0.,1.),(1.,1.)]
#     y = [0.,1.,1.,0.]
#
#     x_train = np.asarray(x)
#     y_train = np.asarray(y)
#     print(x_train)
#     print(y_train)
#     model = createModel((2,),8,1)
#     train(model,x_train,y_train,16,1000,x_train,y_train,'xor.hd5')
#     predict(model,x_train)
#
#
# if __name__ == '__main__':
#     main()

# -----------------------------------------------

'''
exercise2
'''
# #from __future__import print_function
# from tensorflow.python.keras.models import  Sequential, load_model
# from tensorflow.python.keras.layers import Dense, Dropout
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.keras.utils import to_categorical
# from tensorflow.python.keras.datasets import mnist
# import os
#
# def build_model(model_name):
#     if os.path.exists(model_name):
#         print('loading existing model')
#         model = load_model(model_name)
#
#     else:
#         print('making new model')
#         model = Sequential()
#         model.add(Dense(8192,input_shape=(784,),activation='relu'))
#         model.add(Dropout(0))
#         model.add(Dense(4096,activation='relu'))
#         model.add(Dropout(0))
#         model.add(Dense(10,activation='softmax'))
#     return model
#
#
# def train(model,train_x,train_y,epochs,test_x,test_y,model_file):
#     model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
#     print('running for {} epochs'.format(epochs))
#     savemodel = ModelCheckpoint(model_file)
#     stopmodel = EarlyStopping(min_delta=0.001,patience=10)
#     model.fit(x = train_x,y = train_y,shuffle = True,batch_size=60,epochs = epochs,validation_data = (test_x,test_y),callbacks = [savemodel,stopmodel])
#     print('done training now evaluating')
#     loss,acc = model.evaluate(x= test_x,y =test_y)
#     print('final loss is {} and final accuracy is {}'.format(loss,acc))
#
# def load_mnist():
#     (train_x,train_y),(test_x,test_y) = mnist.load_data()
#     train_x = train_x.reshape(train_x.shape[0],784)
#     test_x = test_x.reshape(test_x.shape[0],784)
#
#     train_x = train_x.astype('float32')
#     test_x =test_x.astype('float32')
#
#     train_x /=255.0
#     test_x /= 255.0
#     train_y = to_categorical(train_y,10)
#     test_y = to_categorical(test_y,10)
#
#     return (train_x,train_y),(test_x,test_y)
#
# def main():
#     model = build_model('mnist.hd5')
#     (train_x, train_y), (test_x, test_y) = load_mnist()
#     train(model,train_x,train_y,50,test_x,test_y,'mnist.hd5')
#
# if __name__ == '__main__':
#     main()
#




'''

ex3

'''
#
# from tensorflow.python.keras.models import  Sequential, load_model
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.keras.utils import to_categorical
# from tensorflow.python.keras.datasets import mnist
# import os
#
# MODEL_NAME = 'mnist-cnn.hd5'
#
# def build_model(model_name):
#     if os.path.exists(model_name):
#         print('loading existing model')
#         model = load_model(model_name)
#
#     else:
#
#         model = Sequential()
#
#         model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=(28,28,1),padding='same'))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#         model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#         model.add(Flatten())
#         model.add(Dense(128,activation='relu'))
#         model.add(Dropout(0.3))
#         model.add(Dense(10,activation='softmax'))
#
#
#     return model
#
# def train(model,train_x,train_y,epochs,test_x,test_y,model_file):
#     model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
#     # print('running for {} epochs'.format(epochs))
#     savemodel = ModelCheckpoint(model_file)
#     stopmodel = EarlyStopping(min_delta=0.001,patience=10)
#     print('start training')
#     model.fit(x = train_x,y = train_y,shuffle = True,batch_size=32,epochs = epochs,validation_data = (test_x,test_y),callbacks = [savemodel,stopmodel])
#     print('done training now evaluating')
#     loss,acc = model.evaluate(x= test_x,y =test_y)
#     print('final loss is {} and final accuracy is {}'.format(loss,acc))
#
#
# def load_mnist():
#     (train_x,train_y),(test_x,test_y) = mnist.load_data()
#     train_x = train_x.reshape(train_x.shape[0],28,28,1)
#     test_x = test_x.reshape(test_x.shape[0],28,28,1)
#
#     train_x = train_x.astype('float32')
#     test_x =test_x.astype('float32')
#
#     train_x /=255.0
#     test_x /= 255.0
#     train_y = to_categorical(train_y,10)
#     test_y = to_categorical(test_y,10)
#
#     return (train_x,train_y),(test_x,test_y)
#
#
# def main():
#     (train_x, train_y), (test_x, test_y) = load_mnist()
#     model = build_model(MODEL_NAME)
#     train(model,train_x,train_y,50,test_x,test_y,MODEL_NAME)
#
# if __name__ == '__main__':
#     main()
#



'''

ex4
这个文件还有点问题
'''
# import numpy as np
# from tensorflow.python.keras.models import  Sequential, load_model
# from tensorflow.python.keras.layers import Dense, Dropout, Flatten
# from tensorflow.python.keras.layers import Conv2D,MaxPooling2D
# from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
# from tensorflow.python.keras.utils import to_categorical
#
# from tensorflow.python.keras.datasets import cifar10
# from tensorflow.python.keras.optimizers import SGD,Adadelta
# import os
#
# MODEL_NAME = 'cifar10-cnn.hd5'
#
# def build_model(model_name):
#     if os.path.exists(model_name):
#         print('loading existing model')
#         model = load_model(model_name)
#
#     else:
#
#         model = Sequential()
#
#         model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=(32,32,3),padding='same'))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#         model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
#         model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#         model.add(Flatten())
#         model.add(Dense(128,activation='relu'))
#         model.add(Dropout(0.3))
#         model.add(Dense(10,activation='softmax'))
#     return model
#
# def load_cifar10():
#     (train_x,train_y), (test_x,test_y) = cifar10.load_data()
#     print('train_x shape:',train_x.shape)
#     print('train_x.shape[0] is',train_x.shape[0])
#
#     train_x = train_x.reshape(train_x.shape[0],32,32,3)
#     test_x = test_x.reshape(test_x.shape[0],32,32,3)
#
#     train_x = train_x.astype('float32')
#     test_x =test_x.astype('float32')
#
#     train_x /=255.0
#     test_x /= 255.0
#     train_y = to_categorical(train_y,10)
#     test_y = to_categorical(test_y,10)
#
#     return (train_x,train_y),(test_x,test_y)
#
#
#
# def train(model,train_x,train_y,epochs,test_x,test_y,model_file):
#     model.compile(loss = 'categorical_crossentropy',optimizer ='sgd',metrics = ['accuracy'])
#     # print('running for {} epochs'.format(epochs))
#     savemodel = ModelCheckpoint(model_file)
#     stopmodel = EarlyStopping(min_delta=0.001,patience=10)
#     print('start training')
#     model.fit(x = train_x,y = train_y,shuffle = True,batch_size=256,epochs = epochs,validation_data = (test_x,test_y),callbacks = [savemodel,stopmodel])
#     print('done training now evaluating')
#     loss,acc = model.evaluate(x= test_x,y =test_y)
#     print('final loss is {} and final accuracy is {}'.format(loss,acc))
#
#
#
#
# def main():
#     (train_x, train_y), (test_x, test_y) = load_cifar10()
#     model = build_model(MODEL_NAME)
#     train(model,train_x,train_y,10,test_x,test_y,MODEL_NAME)
#
# if __name__ == '__main__':
#     main()
#

'''
ex5

'''
# import numpy as np
# from tensorflow.python.keras.applications.inception_v3 import InceptionV3
# from tensorflow.python.keras.preprocessing import image
# from tensorflow.python.keras.models import Model,load_model
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import SGD
# import os.path
#
# MODEL_FILE = 'flower.hd5'
#
# def create_model(num_hidden, num_classes):
#     base_model = InceptionV3(include_top = False,weights = 'imagenet')
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(num_hidden,activation='relu')(x)
#     predictions = Dense(num_classes,activation='softmax')(x)
#
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     model = Model(inputs = base_model.input,outputs = predictions)
#     return model
#
# def load_existing(model_file):
#     model = load_model(model_file)
#
#     numlayers = len(model.layers)
#
#     for layer in model.layers[:numlayers-3]:
#         layer.trainable = False
#     for layer in model.layers[numlayers-3:]:
#         layer.trainable = True
#
#     return model
#
# def train(model_file,train_path,validation_path,num_hidden = 200,num_classes=5,steps = 32,num_epochs = 20,save_period=1):
#     if os.path.exists(model_file):
#         print('\n existing model found in {}loading \n'.format(model_file))
#         model = load_existing(model_file)
#
#     else:
#         print('creating a new one')
#         model = create_model(num_hidden,num_classes)
#
#     model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy')
#     checkpoint = ModelCheckpoint(model_file,period=save_period)
#
#     train_datagen = ImageDataGenerator(rescale=1./255,shear_range = 0.2,zoom_range=0.2,horizontal_flip = True)
#     test_datagen = ImageDataGenerator(rescale=1./255)
#     train_generator = train_datagen.flow_from_directory(train_path,target_size=(249,249),batch_size=32,class_mode='categorical')
#     validation_generator = test_datagen.flow_from_directory(validation_path,target_size=(249,249),batch_size=32,class_mode='categorical')
#     model.fit_generator(train_generator,steps_per_epoch=steps,epochs=num_epochs,callbacks = [checkpoint],validation_data=validation_generator,validation_steps=50)
#
#     for layer in model.layers[:249]:
#         layer.trainable = False
#     for layer in model.layers[249:]:
#         layer.trainable = True
#
#     model.compile(optimizer=SGD(lr=0.00001,momentum=0.9),loss = 'categorical_crossentropy')
#     model.fit_generator(train_generator,steps_per_epoch=steps,epochs=num_epochs,callbacks = [checkpoint],validation_data=validation_generator,validation_steps=50)
#
#
# def main():
#     train(MODEL_FILE,train_path='flower_photos',validation_path='flower_photos',steps=5,num_epochs=3)
# #120 10别忘了该回去
# if __name__ == '__main__':
#     main()

#
# import numpy as np
# from tensorflow.python.keras.applications.inception_v3 import InceptionV3
# from tensorflow.python.keras.preprocessing import image
# from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.python.keras.callbacks import ModelCheckpoint
# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.optimizers import SGD
# import os.path
#
# MODEL_FILE = "flowers.hd5"
#
#
# def create_model(num_hidden, num_classes):
#     base_model = InceptionV3(include_top=False, weights='imagenet')
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(num_hidden, activation='relu')(x)
#     predictions = Dense(num_classes, activation='softmax')(x)
#
#     for layer in base_model.layers:
#         layer.trainable = False
#
#     model = Model(inputs=base_model.input, outputs=predictions)
#
#     return model
#
#
# def load_existing(model_file):
#     model = load_model(model_file)
#     numlayers = len(model.layers)
#     for layer in model.layers[:numlayers - 3]:
#         layer.trainable = False
#
#     for layer in model.layers[numlayers - 3:]:
#         layer.trainable = True
#     return model
#
#
# def train(model_file, train_path, validation_path, num_hidden=200, num_classes=5, steps=32, num_epochs=20,
#           save_period=1):
#     if os.path.exists(model_file):
#         print("\n*** Existing model found at {0}.Loading.***\n\n".format(model_file))
#         model = load_existing(model_file)
#     else:
#         print("\n*** Creating new model ***\n\n")
#         model = create_model(num_hidden, num_classes)
#
#     model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#
#     checkpoint = ModelCheckpoint(model_file, period=save_period)
#
#     train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                        shear_range=0.2,
#                                        zoom_range=0.2,
#                                        horizontal_flip=True)
#     test_datagen = ImageDataGenerator(rescale=1. / 225)
#
#     train_generator = train_datagen.flow_from_directory(train_path,
#                                                         target_size=(249, 249),
#                                                         batch_size=32,
#                                                         class_mode='categorical')
#     validation_generator = test_datagen.flow_from_directory(validation_path,
#                                                             target_size=(249, 249),
#                                                             batch_size=32,
#                                                             class_mode='categorical')
#     model.fit_generator(train_generator,
#                         steps_per_epoch=steps,
#                         epochs=num_epochs,
#                         callbacks=[checkpoint],
#                         validation_data=validation_generator,
#                         validation_steps=50)
#
#     for layer in model.layers[:249]:
#         layer.trainable = False
#
#     for layer in model.layers[249:]:
#         layer.trainable = True
#
#     model.compile(optimizer=SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy')
#
#
# def main():
#     train(MODEL_FILE, train_path="flower_photos", validation_path="flower_photos", steps=120, num_epochs=10)
#
#
# if __name__ == "__main__":
#     main()
#























#
# from __future__ import print_function
# from tensorflow.python.keras.models import load_model
# import tensorflow as tf
# import numpy as np
# from PIL import Image
#
# MODEL_NAME = 'flower.hd5'
# dict = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
# graph = tf.get_default_graph()
# def classify(model,image):
#     global graph
#     with graph.as_default():
#         result = model.predict(image)
#         themax = np.argmax(result)
#
#     return (dict[themax],result[0][themax],themax)
# def load_image(image_fname):
#     img = Image.open(image_fname)
#     img = img.resize((249,249))
#     imgarray = np.array(img)/255.0
#     final = np.expand_dims(imgarray,axis=0)
#     return final
#
# def main():
#     model = load_model(MODEL_NAME)
#     img = load_model('tulip2.jpg')
#     label,prob,_ = classify(model,img)
#
#     print('we think with certainty {} that it is {}'.format(prob,label))
#
#
# if __name__ == '__main__':
#     main()
#

































# from __future__ import print_function
# import paho.mqtt.client as mqtt
# import time
#
# USERID = 'sws001'
# PASSWORD = 'persiancat'
# resp_callback = None
#
# def on_connect(client, userdata,flags,rc):
#     print('connected result code is {}'.format(rc))
#     client.subscribe(USERID+"/IMAGE/predict")
#
# def on_message(client,userdata,msg):
#     print('received message from server.',msg.payload)
#     tmp = msg.payload.decode('utf-8')
#     str = tmp.split(":")
#
#     if resp_callback is not None:
#         resp_callback(str[0],float(str[1]),int(str[2]))
#
# def setup():
#     global client
#     client = mqtt.Client(transport='websockets')
#     client.username_pw_set(USERID,PASSWORD)
#     client.on_connect = on_connect
#     #why it's not on_connect()?
#     client.on_message = on_message
#     client.connect('pi.smbox.co',80,30)
#     client.loop_start()
#
# def load_image(filename):
#     with open(filename,'rb') as f:
#         data = f.read()
#     return data
#
# def send_image(filename):
#     img = load_image(filename)
#     client.publish(USERID+'/IMAGE/classify',img)
#
# def resp_handler(label,prob,index):
#     print("\n--response--\n\n")
#     print("label is {}".format(label))
#     print("probability is {}".format(prob))
#     print('index is {}'.format(index))
#
# def main():
#     global resp_callback
#     setup()
#     resp_callback = resp_handler
#     print('waiting 2 seconds before sending')
#     time.sleep(2)
#     print('sending tulip.jpg')
#     send_image('tulip.jpg')
#     print('DONE,waiting 5 seconds before sending')
#     time.sleep(5)
#     print('sending tulip2.jpg')
#     send_image('tulip2.jpg')
#     while True:
#         pass
#
# if __name__ == '__main__':
#     main()
#



#
# from __future__ import print_function
# import paho.mqtt.client as mqtt
# import numpy as np
# from PIL import Image
#
# USERID = 'sws001'
# PASSWORD = 'password'
#
# TMP_FILE = '/tmp/'+USERID+'.jpg'
# dict = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}
#
# def load_image(image):
#     img = Image.open(image)
#     img = img.resize((249,249))
#     imgarray = np.array(img)/255.0
#     final = np.expand_dims(imgarray,axis=0)
#     return final
#
# def classify(imgarray,dict):
#     return dict[4],0.98,4
#
# def on_connect(client,userdata,flags,rc):
#     print('connected with result code {}',format(rc))
#     client.subscribe(USERID+'/IMAGE/classify')
#
# def on_message(client,userdata,msg):
#     print('received messsage. writing to {}'.format(TMP_FILE))
#     img = msg.payload
#
#     with open(TMP_FILE,'wb') as f:
#         f.write(img)
#     imgarray = load_image(TMP_FILE)
#     label,prob,index = classify(imgarray,dict)
#     print('classified as {} with certainty {}'.format(label,prob))
#     client.publish(USERID+'/IMAGE/predict',label+":"+str(prob)+":"+str(index))
#
# def setup():
#     global dict
#     global client
#     client = mqtt.Client(transport='websockets')
#     client.username_pw_set(USERID,PASSWORD)
#     client.on_message = on_message
#     client.on_connect = on_connect
#
#     print('connecting')
#     x = client.connect('pi.smbox.co',80,30)
#     client.loop_start()
#
# def main():
#     setup()
#     while True:
#         pass
#
# if __name__ == '__main__':
#     main()
#

#
#
# from keras.models import Sequential,load_model
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.embeddings import Embedding
# from keras.callbacks import ModelCheckpoint
# from os.path import isfile
# from read_tc import ReadTC
# from constants import *
#
#
# if isfile(filename):
#     print()
#
#
#










































































#
# from captcha.image import  ImageCaptcha
# from PIL import Image
# import random
# import time
# import os
#
# def random_captcha():
#     captcha_text = []
#     for i in range(4):
#         c = str(random.randint(0,5))
#         captcha_text.append(c)
#     return ''.join(captcha_text)
#
# def gen_capthca():
#     image = ImageCaptcha()
#     capthcha_text = random_captcha()
#     capthca_image = Image.open(image.generate(capthcha_text))
#     return capthcha_text,capthca_image
#
# if __name__ == '__main__':
#     count=100
#     path = './captcha_image'
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#     for i in range(count):
#         now = str(int(time.time()))
#         text,image = gen_capthca()
#         filename = text+''+now+'.png'
#         image.save(path+os.path.sep+filename)
#         print('saved {}'.format(filename))
#




from imageai.Detection import ObjectDetection
import os
import time
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()
a = time.time()

custom_objects = detector.CustomObjects(bottle=True)

detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects,input_image=os.path.join(execution_path,'image_bottle2.jpg'),output_image_path=os.path.join(execution_path,'imagenew.jpg'),minimum_percentage_probability=50,box_show=True)
b = time.time()
print('the time is {}'.format(b-a))
print('the direction is {}'.format(detections[0]['direction']))
for eachObject in detections:
    print(eachObject['name']+':'+eachObject['percentage_probability'])





















