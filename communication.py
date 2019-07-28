import serial
import time
import camera  
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)

def handshake():   
    link = 0;
    try:
        while(link == 0):
            print("Send device name to awake...")
            ser.readline();#clear
            user = input() #+ '\n'
            ser.write(user.encode('utf-8'))
#            ser.write("S".encode('utf-8'))
            print("Raspberry Pi send: "+ user)
            print("Waiting for first reply...")
            time.sleep(2)
            response = ser.readline().decode('utf-8')
            if (response == "ack"):
#               ack_message = input()
                link = 1;
                ack_message = "ACK"
                ser.write(ack_message.encode('utf-8'))                
                print("ack received! Ardiuno Online.")
                print("Send message to confirm...")
                print("Raspberry Pi replied.")
                print("Comunication confirm!")
                print("============================")
            else:
                print("Receive: " + response)
                print("Feedback error! Please retry!\n")
                print("===========================")
    except:
        ser.close();
                
#    with bug for overtime


def write():
    user = input("Type the command: ") #+ '\n'
    if (user == 't'):
        camera.start()
    elif (user == 'q'):
        camera.stop()
        print("==End of Photograph==")
    elif (user == 's'):
        camera.start()
        print("==Begin of Photograph==")    
    else:        
        ser.write(user.encode('utf-8'))
#    ser.write("S".encode('utf-8'))
        print("Raspberry Pi send: "+ user)

def send(text):
    user = text
    ser.write(user.encode('utf-8'))
#    ser.write("S".encode('utf-8'))
    print("Raspberry Pi send: "+ user)
 
    
