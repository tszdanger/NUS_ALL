from __future__ import print_function
import sys
import pygame
import communication as cm
import camera
from pygame.locals import *

from imageai.Detection import ObjectDetection

from tensorflow.python.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time
#MODEL_NAME = 'morecatsince_v5.hd5'
MODEL_NAME = '/home/pi/Desktop/v4.hd5'
dict = {0: 'Maine_Coon', 2: 'Singapura', 3: 'Turkish_Van', 1: 'Ocelot'}
graph = tf.get_default_graph()


def classify(model, image):
    global graph
    with graph.as_default():
        result = model.predict(image)
        themax = np.argmax(result)

    return (dict[themax], result[0][themax], themax)


def load_image(image_fname):
    img = Image.open(image_fname)
    img = img.resize((249, 249))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final




def control():
    global num_seed
    num_seed = 0
    model = load_model(MODEL_NAME)
    #-----------
    imagede = '/home/pi/Desktop/photos/default.jpg'
    img_default = load_image(imagede)
    classify(model, img_default)

    ##初始化
    pygame.init()
    ##变量存放处

    size = width, height = 300, 200
    bgColor = (0, 0, 0)

    ##設置界面寬高

    screen = pygame.display.set_mode(size)

    ##設置標題

    pygame.display.set_caption("Team 1 Monitor")

    ##要在Pygame中使用文本，必须创建Font对象

    ##第一个参数指定字体 ，第二个参数指定字体大小

    font = pygame.font.Font(None, 20)

    ##调用get_linesize()方法获得每行文本的高度

    line_height = font.get_linesize()
    position = 0
    screen.fill(bgColor)

    ##创建一个存放的文本TXT

    # f = open("record.txt",'w')

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # 關閉文件
                # f.close()
                sys.exit()
            # print('GG\n')
            if event.type == pygame.KEYDOWN:
                # f.write(str(event) + '\n')
                if event.key == K_w:
                    # print('w\n')
                    cm.send('#W')
                elif event.key == K_s:
                    cm.send('#S')
                if event.key == K_j:
                    # print('w\n')
                    cm.send('#w')
                elif event.key == K_k:
                    cm.send('#s')

                elif event.key == K_d:
                    cm.send('#D')
                elif event.key == K_a:
                    cm.send('#A')

                elif event.key == K_x:
                    cm.send('#x')
                elif event.key == K_b:
                    cm.send('#b')
                # --------------------------------------------

                elif event.key == K_p:
                    camera.stop()
                    imagepath = '/home/pi/Desktop/photos/' + str(num_seed) + '.jpg'
                    img = load_image(imagepath)
                    label, prob, _ = classify(model, img)
                    print('we think image name:{} with certainty {} that it is {}'.format(imagepath, prob, label))
                        
                # ------------------------------
                # 目标跟随，返回
                #  hd5文件请放在执行文件目录下，输入输出在photos文件夹
                elif event.key == K_g:
                    camera.stop()
                    imagepath = '/home/pi/Desktop/photos/' + str(num_seed) + '.jpg'
                    outputpath = '/home/pi/Desktop/photos/' + str(num_seed) + 'new.jpg'
                    execution_path = os.getcwd()
                    detector = ObjectDetection()
                    detector.setModelTypeAsRetinaNet()
                    detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
                    detector.loadModel()
                    a = time.time()

                    custom_objects = detector.CustomObjects(bottle=True)

                    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects,
                                                                       input_image=imagepath,
                                                                       output_image_path=outputpath,
                                                                       minimum_percentage_probability=50, box_show=True)
                    b = time.time()
                    print('the time is {}'.format(b - a))
                    print('the direction is {}'.format(detections[0]['direction']))
                    for eachObject in detections:
                        print(eachObject['name'] + ':' + eachObject['percentage_probability'])



                elif event.key == K_t:
                    num_seed = camera.capture(num_seed)


                elif event.key == K_q:
                    camera.stop()
                    print("==End of Photograph==")
                elif event.key == K_o:
                    camera.start()
                    print("==Begin of Photograph==")
                elif event.key == K_r:
                    camera.record()
                    # render()将文本渲染成Surface对象
                # 第一个参数是带渲染的文本
                # 第二个参数指定是否消除锯齿
                # 第三个参数指定文本的颜色
                screen.blit(font.render(str(event), True, (0, 255, 0)), (0, position))
                position += line_height
                if position >= height:
                    position = 0
                    screen.fill(bgColor)
                pygame.display.flip()

# 上面这几段代码在干嘛
