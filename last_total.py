from __future__ import print_function
import paho.mqtt.client as mqtt
import time

USERID ="sws001"
PASSWORD = "persiancat"

resp_callback = None

import wave
import numpy as np
import os
from keras.models import load_model


import pyaudio
import wave

from PIL import Image

from imageai.Detection import ObjectDetection
import os
import time


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"

TMP_FILE = "C:\\Users\\skywf\\Desktop\\docker_image.jpg"
dict = {0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}


'''

语音识别用的函数

'''


def on_connect(client, userdata, flags, rc):
    print("Connected. Result code is %d." % (rc))
    client.subscribe(USERID + "/IMAGE/predict")


def on_message(client, userdata, msg):
    print("Received message from server.", msg.payload)

    tmp = msg.payload.decode("utf-8")
    str = tmp.split(":")

    if resp_callback is not None:
        resp_callback(Str[0], float(str[1]), int(str[2]))


def setup():
    global client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("192.168.43.70", 1883, 30)
    client.loop_start()


def load_image(filename):
    with open(filename, "rb") as f:
        data = f.read()
    return data


def send_image(filename):
    img = load_image(filename)
    client.publish(USERID + "/IMAGE/classify", img)


def send_info(label):
    client.publish(USERID + "/IMAGE/classify", label)


def resp_handler(label, prob, index):
    print("\n --Response --\n\n")
    print("Label: %s" % (label))
    print("Probability:%3.4f" % (prob))
    print("Index:%d" % (index))



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

    接受图片用的函数(全部加上_1区分)
    和传回

'''
def tell_dire():
    # 路径直接写死了C:\\Users\\skywf\\Desktop\\docker_image.jpg图片直接输出到桌面
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
    detector.loadModel()
    # a = time.time()
    custom_objects = detector.CustomObjects(cat =True)
    detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects,input_image='C:\\Users\\skywf\\Desktop\\docker_image.jpg',output_image_path='C:\\Users\\skywf\\Desktop\\imagenew.jpg',minimum_percentage_probability=50,box_show=True)
    # b = time.time()
    # print('the time is {}'.format(b-a))
    print('the direction is {}'.format(detections[0]['direction']))
    for eachObject in detections:
        print(eachObject['name']+':'+eachObject['percentage_probability'])
    return detections[0]['direction']

def load_image_1(image):
    img = Image.open(image)
    img = img.resize((249,249))
    imgarray = np.array(img)/255.0
    final = np.expand_dims(imgarray,axis=0)
    return final

def classify(imgarray,dict):
    return dict[4],0.98,4

def on_connect_1(client,userdata,flags,rc):
    print('connected with result code {}'.format(rc))
    client.subscribe(USERID+'/IMAGE/classify')

def on_message_1(client,userdata,msg):
    print('received messsage. writing to {}'.format(TMP_FILE))
    img = msg.payload
    #print(img)
    with open(TMP_FILE,'wb') as f:
        print("1")
        f.write(img)

    imgarray = load_image_1(TMP_FILE)
    print("2")
    # label,prob,index = classify(imgarray,dict)
    # print('Classified as {} with certainty {}'.format(label,prob))
    label = 'the direction '
    prob = 'is'
    direction = tell_dire()
    client.publish(USERID+'/IMAGE/predict',label+":"+prob+":"+direction)

def setup_1():
    global dict
    global client
    client = mqtt.Client()
    client.on_connect = on_connect_1
    client.on_message = on_message_1

    print('Connecting.')
    x = client.connect("192.168.43.70",1883,30)
    client.loop_start()




def main():
    a = input('please tell me what you want 1.语音识别 2.接受图片+框图发送中心 ')
    if (a == '1'):
        # 先录音
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
        wavs.append(get_wav_mfcc("D:\\Github\\kerasTfPoj\\kerasTfPoj\\ASR\\output.wav"))#再读
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
        global resp_callback
        setup()
        res_callback = resp_handler
        label = name[ind]
        send_info(label)
        print("instruction has been sent")

    elif (a =='2'):
        setup_1()


if __name__ == '__main__':
        main()