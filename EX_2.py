from __future__ import print_function
from tensorflow.python.keras.models import  Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import mnist
import os

def build_model(model_name):
    if os.path.exists(model_name):
        print('loading existing model')
        model = load_model(model_name)

    else:
        print('making new model')
        model = Sequential()
        model.add(Dense(1024,input_shape=(784,),activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10,activation='softmax'))
    return model


def train(model,train_x,train_y,epochs,test_x,test_y,model_file):
    model.compile(loss = 'categorical_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
    print('running for {} epochs'.format(epochs))
    savemodel = ModelCheckpoint(model_file)
    stopmodel = EarlyStopping(min_delta=0.001,patience=10)
    model.fit(x = train_x,y = train_y,shuffle = True,batch_size=60,epochs = epochs,validation_data = (test_x,test_y),callbacks = [savemodel,stopmodel])
    print('done training now evaluating')
    loss,acc = model.evaluate(x= test_x,y =test_y)
    print('final loss is {} and final accuracy is {}'.format(loss,acc))

def load_mnist():
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    train_x = train_x.reshape(train_x.shape[0],784)
    test_x = test_x.reshape(test_x.shape[0],784)

    train_x = train_x.astype('float32')
    test_x =test_x.astype('float32')

    train_x /=255.0
    test_x /= 255.0
    train_y = to_categorical(train_y,10)
    test_y = to_categorical(test_y,10)

    return (train_x,train_y),(test_x,test_y)

def main():
    model = build_model('mnist.hd5')
    (train_x, train_y), (test_x, test_y) = load_mnist()
    train(model,train_x,train_y,50,test_x,test_y,'mnist.hd5')

if __name__ == '__main__':
    main()
