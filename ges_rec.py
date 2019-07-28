import cv2



def binaryMask(frame,x0,y0,width,height):
    minValue = 70
    cv.rectangle(frame,(x0,y0),(x0+width,y0+width),(0,255,0),1)
    roi = frame[y0:y0+height,x0,x0+width]

print('helloD:\wechat_helper\ges_rec.py')