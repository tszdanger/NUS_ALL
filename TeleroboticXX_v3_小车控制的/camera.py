import signal
import sys
import time
from picamera import PiCamera, Color
from time import sleep

def emergency(signal, frame):
    print('Something went wrong!')
    sys.exit(0)

signal.signal(signal.SIGINT, emergency)

demoCamera = PiCamera()
global recording 

def start():
    demoCamera.resolution = (640,480)
    demoCamera.framerate = 30
    demoCamera.start_preview(alpha = 220)
    demoCamera.annotate_background = Color('white')
    demoCamera.annotate_foreground = Color('red')
    demoCamera.annotate_text = "sTeam1_SWS3009_2019"
    global recording
    recording = 0
# def capture():
    # frame = str(time.strftime("%Y%m%d%H%M%S",time.localtime()))
    # demoCamera.capture('/home/pi/Desktop/photos/%s.jpg' % frame)


def capture(num_seed):
    # frame = str(time.strftime("%Y%m%d%H%M%S",time.localtime()))
    num_seed +=1
    frame = num_seed
    demoCamera.capture('/home/pi/Desktop/photos/%s.jpg' % frame)
    return num_seed



def record():
    global recording
    if (recording == 0):
        frame = str(time.strftime("%Y%m%d%H%M%S",time.localtime()))
        demoCamera.start_recording('/home/pi/Desktop/photos/%s.h264' % frame)
        recording = 1
    else:
        demoCamera.stop_recording()
        recording = 0



def stop():
    demoCamera.stop_preview()

