'''

    这是可以用的!

'''

from __future__ import print_function
from socket import *
import os

import paho.mqtt.client as mqtt
import time

import wave
import numpy as np
from keras.models import load_model


import pyaudio
from PIL import Image
from imageai.Detection import ObjectDetection
import sys

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"

TMP_FILE = "C:\\Users\\skywf\\Desktop\\docker_image.jpg"
dict = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

'''
语音识别函数

'''
def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

    ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的【因为训练文件全部是1秒钟长度，16000帧的，所以这里需要把每个语音文件的长度处理成一样的】
    data = list(np.array(waveData[0]))
    print(len(data))
    count1 = 0
    while len(data)>16000:
        count1 +=1
        del data[len(waveData[0])-2*count1]
        del data[count1-1]
    # print(len(data))
    while len(data)<16000:
        data.append(0)
    # print(len(data))

    data=np.array(data)
    # 平方之后，开平方，取正数，值的范围在  0-1  之间
    data = data ** 2
    data = data ** 0.5
    return data
'''
    路径识别函数
'''
def tell_dire():
    # 路径直接写死了C:\\Users\\skywf\\Desktop\\docker_image.jpg图片直接输出到桌面
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
    detector.loadModel()
    # a = time.time()
    custom_objects = detector.CustomObjects(bottle =True)
    detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects,input_image='C:\\Users\\skywf\\Desktop\\docker_image.jpg',output_image_path='C:\\Users\\skywf\\Desktop\\imagenew.jpg',minimum_percentage_probability=50,box_show=True)
    # b = time.time()
    # print('the time is {}'.format(b-a))
    # print('the direction is {}'.format(detections[0]['direction']))
    for eachObject in detections:
        print(eachObject['name']+':'+eachObject['percentage_probability'])
    return detections[0]['direction']

def main():
    a = input('please tell me what you want 1.语音识别 2.接受图片+框图发送中心 ')
    if (a == '1'):
        serverName = "192.168.43.70"
        serverport = 12000
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((serverName, serverport))
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        model = load_model('asr_model_weights.h5')  # 加载训练模型
        wavs = []
        wavs.append(get_wav_mfcc("D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"))  # 再读
        X = np.array(wavs)
        print(X.shape)
        result = model.predict(X[0:1])[0]  # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
        print("识别结果", result)
        #  因为在训练的时候，标签集的名字 为：  0：go   1：stop    0 和 1 是下标
        name = ["go", "stop"]  # 创建一个跟训练时一样的标签集
        ind = 0  # 结果中最大的一个数
        for i in range(len(result)):
            if result[i] > result[ind]:
                ind = 1
        print("识别的语音结果是：", name[ind])
        # 再传label

        label = name[ind]
        clientSocket.send(label.encode())

        clientSocket.close()



    elif (a=='2'):
        serverPort = 12000
        serverSocket = socket(AF_INET, SOCK_STREAM)
        serverSocket.bind(('', serverPort))
        serverSocket.listen(1)
        connectionSocket, addr = serverSocket.accept()
        f = open("C:\\Users\\skywf\\Desktop\\docker_image.jpg", "wb")
        # a = time.time()
        while True:
            data = connectionSocket.recv(1024)
            if not data:
                break
            f.write(data)
            # b = time.time()
            # if (b - a - 4) > 0:
            #     break
        print("image has been received")
        f.close()

        direction = tell_dire()
        print('!!!direction is {}'.format(direction))
        # connectionSocket.send(direction.encode())
        connectionSocket.close()


#实在是搞不定那个东西所以重新发一遍
        serverName = "192.168.43.70"
        serverport = 12000
        clientSocket = socket(AF_INET, SOCK_STREAM)
        clientSocket.connect((serverName, serverport))
        clientSocket.send(direction.encode())
        clientSocket.close()





if __name__ == '__main__':
        main()