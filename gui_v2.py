from __future__ import print_function
import tkinter as tk

import pyaudio
import wave

import numpy as np
import os
from keras.models import load_model

import paho.mqtt.client as mqtt
import time

from PIL import Image

import imagedetect_v1


USERID = "sws001"
PASSWORD = "persiancat"

resp_callback = None


TMP_FILE = "C:\\Users\\skywf\\Desktop\\docker_image.jpg"
dict = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

def on_connect_1(client, userdata, flags, rc):
    print("Connected. Result code is %d." % (rc))
    client.subscribe(USERID + "/IMAGE/predict")


def on_message_1(client, userdata, msg):
    print("Received message from server.", msg.payload)

    tmp = msg.payload.decode("utf-8")
    str = tmp.split(":")

    if resp_callback is not None:
        resp_callback(Str[0], float(str[1]), int(str[2]))


def setup_1():
    global client
    client = mqtt.Client()
    client.on_connect = on_connect_1
    client.on_message = on_message_1
    client.connect("192.168.43.70", 1883, 30)
    client.loop_start()


def load_image_1(filename):
    with open(filename, "rb") as f:
        data = f.read()
    return data


def send_image_1(filename):
    img = load_image(filename)
    client.publish(USERID + "/IMAGE/classify", img)


def send_info_1(label):
    client.publish(USERID + "/IMAGE/classify", label)


def resp_handler_1(label, prob, index):
    print("\n --Response --\n\n")
    print("Label: %s" % (label))
    print("Probability:%3.4f" % (prob))
    print("Index:%d" % (index))

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"
def get_wav_mfcc(wav_path):
    f = wave.open(wav_path,'rb')
    params = f.getparams()
    # print("params:",params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)#读取音频，字符串格式
    waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
    waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
    waveData = np.reshape(waveData,[nframes,nchannels]).T
    f.close()

    ### 对音频数据进行长度大小的切割，保证每一个的长度都是一样的【因为训练文件全部是1秒钟长度，16000帧的，所以这里需要把每个语音文件的长度处理成一样的】
    data = list(np.array(waveData[0]))
    count1 = 0
    while len(data) > 16000:
        count1 += 1
        del data[len(waveData[0]) - 2 * count1]
        del data[count1 - 1]
    # print(len(data))
    while len(data)<16000:
        data.append(0)
    # print(len(data))

    data=np.array(data)
    # 平方之后，开平方，取正数，值的范围在  0-1  之间
    data = data ** 2
    data = data ** 0.5
    return data


window = tk.Tk()
window.title('for_speaking')
window.geometry('500x400')

var1 = tk.StringVar()
var2 = tk.StringVar()

l = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var1)
l.pack()
l1 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var2)
l1.pack()

on_hit = False


def hit():
    global on_hit
    if on_hit==False:
        on_hit=True
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")
        var1.set('录制时长'+str(RECORD_SECONDS)+'s')
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
        var2.set('录制完成')

    else:
        on_hit=False
        # var1.set('no')

b = tk.Button(window,text = '点击开始录制',font = ('Arial',12),width=10,height = 1,command = hit)
b.pack()

var3 = tk.StringVar()
var4 = tk.StringVar()

l2 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var3)
l2.pack()
l3 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var4)
l3.pack()

pre_hit = False
lable_instru = 'Nothing'

def pre():
    global pre_hit
    global lable_instru
    model = load_model('D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\asr_model_weights.h5')  # 加载训练模型
    wavs = []
    if pre_hit==False:
        pre_hit=True

        print("* predicting")
        # var1.set('录制时长'+str(RECORD_SECONDS)+'s')
        wavs.append(get_wav_mfcc("D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"))
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
        lable_instru = name[ind]

    else:
        pre_hit=False
        # var1.set('no')

c = tk.Button(window,text = '点击开始预测',font = ('Arial',12),width=10,height = 1,command = pre)
c.pack()



var5 = tk.StringVar()
var6 = tk.StringVar()

l4 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var5)
l4.pack()
l5 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var6)
l5.pack()

def trans():
    global resp_callback
    setup_1()
    res_callback = resp_handler
    # label = "go"
    send_info_1(lable_instru)
    print("instruction has been sent")

d = tk.Button(window,text = '点击开始传送',font = ('Arial',12),width=10,height = 1,command = trans)
d.pack()


var7 = tk.StringVar()

l6 = tk.Label(window,bg = 'green',fg = 'yellow',font=('Arial',12),width=25,textvariable = var7)
l6.pack()


def load_image(image):
    img = Image.open(image)
    img = img.resize((249,249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray,axis=0)
    return final

def classify(imgarray,dict):
    return dict[4],0.98,4

def on_connect(client,userdata,flags,rc):
    print('connected with result code {}'.format(rc))
    client.subscribe(USERID+'/IMAGE/classify')

def on_message(client,userdata,msg):
    print('received messsage. writing to {}'.format(TMP_FILE))
    img = msg.payload
    #print(img)
    with open(TMP_FILE,'wb') as f:
        print("1")
        f.write(img)

    # imgarray = load_image(TMP_FILE)
    # print("2")
    # label,prob,index = classify(imgarray,dict)
    # print('Classified as {} with certainty {}'.format(label,prob))
    # client.publish(USERID+'/IMAGE/predict',label+":"+str(prob)+":"+'go')




def setup():
    global dict
    global client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print('Connecting.')
    x = client.connect("192.168.43.70",1883,30)
    client.loop_start()




def wait_now():
    setup()
    imagedetect_v1.tell_dire()

e = tk.Button(window,text = '点击开始等待图片',font = ('Arial',12),width=10,height = 1,command = wait_now)
e.pack()





window.mainloop()