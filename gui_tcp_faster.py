
from __future__ import print_function
from socket import *
import os
import tkinter as tk

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
    custom_objects = detector.CustomObjects(book = True,cell_phone= True)
    detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects,input_image='C:\\Users\\skywf\\Desktop\\docker_image.jpg',output_image_path='C:\\Users\\skywf\\Desktop\\imagenew.jpg',minimum_percentage_probability=40,box_show=True)
    print(len(detections))
    if(len(detections) == 0):
        return 'forward'
    # b = time.time()
    # print('the time is {}'.format(b-a))
    # print('the direction is {}'.format(detections[0]['direction']))
    else:
        for eachObject in detections:
            print(eachObject['name']+':'+eachObject['percentage_probability'])
        return detections[0]['direction']
# yuyin仅仅支持go stop
def yuyin():
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

# yuyin1支持happy sheila
def yuyin1():
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
    model = load_model('asr_model_2_sheila.h5')  # 加载训练模型
    wavs = []
    wavs.append(get_wav_mfcc("D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"))  # 再读
    X = np.array(wavs)
    print(X.shape)
    result = model.predict(X[0:1])[0]  # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
    print("识别结果", result)
    #  因为在训练的时候，标签集的名字 为：  0：go   1：stop    0 和 1 是下标
    name = ["happy", "sheila"]  # 创建一个跟训练时一样的标签集
    ind = 0  # 结果中最大的一个数
    for i in range(len(result)):
        if result[i] > result[ind]:
            ind = 1
    print("识别的语音结果是：", name[ind])
    # 再传label

    label = name[ind]
    clientSocket.send(label.encode())

    clientSocket.close()
# yuyin2支持left right

def yuyin2():
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
    model = load_model('asr_model_2_lstm.h5')  # 加载训练模型
    wavs = []
    wavs.append(get_wav_mfcc("D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"))  # 再读
    X = np.array(wavs)
    print(X.shape)
    result = model.predict(X[0:1])[0]  # 识别出第一张图的结果，多张图的时候，把后面的[0] 去掉，返回的就是多张图结果
    print("识别结果", result)
    #  因为在训练的时候，标签集的名字 为：  0：right   1：left    0 和 1 是下标
    name = ['right','left']  # 创建一个跟训练时一样的标签集
    ind = 0  # 结果中最大的一个数
    for i in range(len(result)):
        if result[i] > result[ind]:
            ind += 1
    print("识别的语音结果是：", name[ind])
    # 再传label

    label = name[ind]
    clientSocket.send(label.encode())

    clientSocket.close()

def tupian():
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

    # 实在是搞不定那个东西所以重新发一遍
    serverName = "192.168.43.70"
    serverport = 12000
    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName, serverport))
    clientSocket.send(direction.encode())
    clientSocket.close()

window = tk.Tk()
window.title('for_speaking')
window.geometry('500x400')
var1 = tk.StringVar()
var2 = tk.StringVar()
l = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var1)
l.pack()
l1 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var2)
l1.pack()
b = tk.Button(window,text = '语音识别go/stop',font = ('Arial',12),width=15,height = 1,command = yuyin)
b.pack()

var5 = tk.StringVar()
var6 = tk.StringVar()
l4 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var5)
l4.pack()
l5 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var6)
l5.pack()
d = tk.Button(window,text = '语音happy/shelina',font = ('Arial',12),width=15,height = 1,command = yuyin1)
d.pack()

var7 = tk.StringVar()
var8 = tk.StringVar()
l6 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var7)
l6.pack()
l7 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=50,textvariable = var8)
l7.pack()
e = tk.Button(window,text = '语音识别right/left',font = ('Arial',12),width=15,height = 1,command = yuyin2)
e.pack()


var3 = tk.StringVar()
var4 = tk.StringVar()
l2 = tk.Label(window,bg = 'grey',fg = 'yellow',font=('Arial',12),width=50,textvariable = var3)
l2.pack()
l3 = tk.Label(window,bg = 'grey',fg = 'yellow',font=('Arial',12),width=50,textvariable = var4)
l3.pack()

c = tk.Button(window,text = '图片处理',font = ('Arial',12),width=15,height = 1,command = tupian)
c.pack()


# ----------------------------放飞自我-----------------------
# def print_selection(v):
#     l.config(text='you have selected ' + v)
# # 第5步，创建一个尺度滑条，长度200字符，从0开始10结束，以2为刻度，精度为0.01，触发调用print_selection函数
# s = tk.Scale(window, label='目前处理', from_=0, to=1, orient=tk.HORIZONTAL, length=200, showvalue=0,tickinterval=1, resolution=0.01, command=print_selection)
# s.pack()


#
# tk.Label(window, text='处理进度:', ).place(x=0, y=260)
# canvas = tk.Canvas(window, width=465, height=22, bg="white")
# canvas.place(x=50, y=260)
#
#
# # 显示下载进度
# def progress():
#     # 填充进度条
#     fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="green")
#     x = 500  # 未知变量，可更改
#     n = 465 / x  # 465是矩形填充满的次数
#     for i in range(x):
#         n = n + 465 / x
#         canvas.coords(fill_line, (0, 0, n, 60))
#         window.update()
#         time.sleep(0.02)  # 控制进度条流动的速度
#
#     # 清空进度条
#     fill_line = canvas.create_rectangle(1.5, 1.5, 0, 23, width=0, fill="white")
#     x = 500  # 未知变量，可更改
#     n = 465 / x  # 465是矩形填充满的次数
#
#     for t in range(x):
#         n = n + 465 / x
#         # 以矩形的长度作为变量值更新
#         canvas.coords(fill_line, (0, 0, n, 60))
#         window.update()
#         time.sleep(0)  # 时间为0，即飞速清空进度条
#
#
# btn_download = tk.Button(window, text='启动进度条', command=progress)
# btn_download.place(x=400, y=280)

canvas = tk.Canvas(window, bg='green', height=300, width=500)
# 说明图片位置，并导入图片到画布上
image_file = tk.PhotoImage(file='kuaile.gif')  # 图片位置（相对路径，与.py文件同一文件夹下，也可以用绝对路径，需要给定图片具体绝对路径）
image = canvas.create_image(300, 500, anchor='n',image=image_file)

#---------------------------------------------------
window.mainloop()



