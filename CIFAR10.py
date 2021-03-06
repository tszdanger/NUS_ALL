from __future__ import print_function
from tensorflow.python.keras.models import  Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar10
import os

MODEL_NAME = 'cifar-cnn.hd5'

def build_model(model_name):
    if os.path.exists(model_name):
        print('loading existing model')
        model = load_model(model_name)

    else:

        model = Sequential()

        model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=(32,32,1),padding='same'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=2))
        model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=2))
        model.add(Flatten())
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10,activation='softmax'))


    return model

def train(model,train_x,train_y,epochs,test_x,test_y,model_file):
    model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
    # print('running for {} epochs'.format(epochs))
    savemodel = ModelCheckpoint(model_file)
    stopmodel = EarlyStopping(min_delta=0.001,patience=10)
    print('start training')
    model.fit(x = train_x,y = train_y,shuffle = True,batch_size=32,epochs = epochs,validation_data = (test_x,test_y),callbacks = [savemodel,stopmodel])
    print('done training now evaluating')
    loss,acc = model.evaluate(x= test_x,y =test_y)
    print('final loss is {} and final accuracy is {}'.format(loss,acc))


def load_mnist():
    (train_x,train_y),(test_x,test_y) = cifar10.load_data()
    train_x = train_x.reshape(train_x.shape[0],32,32,3)
    test_x = test_x.reshape(test_x.shape[0],32,32,3)

    train_x = train_x.astype('float32')
    test_x =test_x.astype('float32')

    train_x /=255.0
    test_x /= 255.0
    train_y = to_categorical(train_y,10)
    test_y = to_categorical(test_y,10)

    return (train_x,train_y),(test_x,test_y)


def main():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    model = build_model(MODEL_NAME)
    train(model,train_x,train_y,50,test_x,test_y,MODEL_NAME)

if __name__ == '__main__':
    main()