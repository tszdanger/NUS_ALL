


from imageai.Detection import ObjectDetection
import os
import time


def tell_dire():
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
    a = time.time()

    detector.loadModel(detection_speed='fastest')
    b = time.time()

    # custom_objects = detector.CustomObjects(bottle=True)
    # detections = detector.detectCustomObjectsFromImage(custom_objects = custom_objects,input_image=os.path.join(execution_path,'image_cat.jpg'),output_image_path=os.path.join(execution_path,'imagenew.jpg'),minimum_percentage_probability=50,box_show=True)

    detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path,'image_cat.jpg'),output_image_path=os.path.join(execution_path,'imagenew.jpg'),minimum_percentage_probability=50,box_show=True)

    print('the time is {}'.format(b-a))
    # print(len(detections))
    # if(len(detections)==0):
    #     detections[0]['direction'] = 'Forward'
    # else:
    print('the direction is {}'.format(detections[0]['direction']))
    print(type(detections[0]['direction']))
    for eachObject in detections:
        print(eachObject['name']+':'+eachObject['percentage_probability'])

if __name__ == '__main__':
    tell_dire()