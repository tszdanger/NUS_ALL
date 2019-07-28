'''

这是接受图片的
超过两秒钟就结束发送
'''


from socket import *
import os,sys
import time
serverPort=12000
serverSocket=socket(AF_INET,SOCK_STREAM)
serverSocket.bind(('',serverPort))
serverSocket.listen(1)
connectionSocket,addr=serverSocket.accept()
f=open("D:\\wechat_helper\\image.jpg","wb")
# a = time.time()
while True:
      data=connectionSocket.recv(1024)
      if not data:
          break
      f.write(data)
      # b=time.time()
      # print(b-a)
      # if (b-a-10)>0:
      #     break
print("image has been received")
f.close()
connectionSocket.close()





