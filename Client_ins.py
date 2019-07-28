from socket import *
import os
serverName="192.168.43.70"
serverport=12000
clientSocket=socket(AF_INET,SOCK_STREAM)
clientSocket.connect((serverName,serverport))
sentence = input("Type in something:")
clientSocket.send(sentence.encode())

clientSocket.close()
