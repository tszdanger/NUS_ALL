from socket import *
import os,sys
serverPort=12000
serverSocket=socket(AF_INET,SOCK_STREAM)
serverSocket.bind(('',serverPort))
serverSocket.listen(1)
print('The server is ready to receive command.')
while 1:
    connectionSocket,addr=serverSocket.accept()
    recvs=connectionSocket.recv(1024)
    msg=recvs.decode()
    print(msg)
    sentence="something"
    connectionSocket.send(msg.encode())
    connectionSocket.close()






