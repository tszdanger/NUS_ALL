import sys
import pygame
import communication as cm
import camera
from pygame.locals import *

from __future__ import print_function
from tensorflow.python.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import os

MODEL_NAME = 'nasnet_cats.hd5'
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
    img = img.resize((331, 331))
    imgarray = np.array(img) / 255.0
    final = np.expand_dims(imgarray, axis=0)
    return final




def control():
    global num_seed
    num_seed = 0
    model = load_model(MODEL_NAME)

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
                #  这下面的代码是云加的,想法是按下p键开始预测,与之对应的向拍照传入种子来编号
                elif event.key == K_p:
                    imagepath = '/home/pi/Desktop/photos/' + str(num_seed) + '.jpg'

                    img = load_image(imagepath)
                    label, prob, _ = classify(model, img)
                    print('we think image name:{} with certainty {} that it is {}'.format(imagepath, prob, label))

                # ---------------------------------------------
                # ---------------------------------------------
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
