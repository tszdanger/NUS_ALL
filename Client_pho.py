from socket import *
import os
serverName="192.168.43.168"
serverport=12000
clientSocket=socket(AF_INET,SOCK_STREAM)
clientSocket.connect((serverName,serverport))
sentence = input("Type in something:")
f=open("classPhoto.jpg","rb")
while True:
    data=f.read(1024)
    if not data:
        break
    clientSocket.send(data)
print("image has been sent")
f.close()

#clientSocket.send(sentence.encode())
rep=clientSocket.recv(2048)
print('From Server:\n',rep.decode())
clientSocket.close()
