import numpy as np
import cv2
import time
import math


def ecl_dist(pt1,pt2):
    dist = math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
    return int(dist)


def rel_distance(image,pts):
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            rel_dist = ecl_dist(pts[i],pts[j])
            rel_distance = round(rel_dist/24,2)
            if rel_distance <= 6.0: #distance in meter
                center1 = (tuple(pts[i][:2]))
                center2 = (tuple(pts[j][:2]))
                axesLength1 = (6000//pts[i][2], 15000//pts[i][2])
                axesLength2 = (6000//pts[j][2], 15000//pts[j][2])
                cv2.ellipse(image, center1, axesLength1,0, 0, 360,(0,0,255), 2)
                cv2.ellipse(image, center2, axesLength2,0, 0, 360,(0,0,255), 2)
                cv2.putText(image,'Not Safe',(pts[j][0],pts[j][1]),font,0.8,(84,213,252),1)


path = 'video/test.mp4'

net = cv2.dnn.readNetFromDarknet('yolov4-safe.cfg','yolov4-safe.weights')


with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

font = cv2.FONT_HERSHEY_COMPLEX_SMALL



class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        t = time.time()
        ret, frame = self.video.read()
        image = cv2.resize(frame, (1000, 560), interpolation=cv2.INTER_NEAREST)
        if ret==True:
            cv2.rectangle(image,(0,0),(155,50),(119,117,7),cv2.FILLED)
            cv2.putText(image,'FPS:',(18,35),font,1.4,(255,255,255),2)
            try:
                ht, wt, _ = image.shape
                blob = cv2.dnn.blobFromImage(image,1/255,(416,416),(0,0,0),swapRB=True,crop = False)
                net.setInput(blob)
                last_layer = net.getUnconnectedOutLayersNames()
                layer_out = net.forward(last_layer)
                boxes = []
                confidences = []
                cls_ids = []
                pts = []
                for output in layer_out:
                    for detection in output:
                        score = detection[5:]
                        clsid = np.argmax(score)
                        conf = score[clsid]
                        if (clsid==0) and (conf > 0.3): # detecting person only
                            centreX, centreY = int(detection[0]*wt), int(detection[1]*ht)
                            w,h = int(detection[2]*wt), int(detection[3]*ht)
                            x,y = int(centreX - w/2), int(centreY - h/2)
                            boxes.append([x,y,w,h])
                            confidences.append((float(conf)))
                            cls_ids.append(clsid)


                indexes = cv2.dnn.NMSBoxes(boxes,confidences,.3,.2)
                colors = np.random.uniform(0,255,size = (len(boxes),2))
                for i in indexes.reshape(-1):
                    x,y,w,h = boxes[i]
                    z = int(30000/h) #depth factor
                    cpt = [int(x+w/2),int(y+h/2),z]
                    pts.append(cpt)
                    color = colors[i]
                    label = str(classes[cls_ids[i]])
                    text = label
                    cv2.rectangle(image,(x,y),(x+62,y-16),(119,117,7),cv2.FILLED)
                    cv2.line(image,(x,y-18),(x,y+h),(119,117,7),2)
                    cv2.putText(image,text,(x+2,y-5),font,0.7,(255,255,255),1)
                rel_distance(image,pts)
                t2 = time.time()
                fps = round(1/(t2-t))
                cv2.putText(image,f'{fps}',(95,35),font,1.5,(255,255,255),2)
                ret, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()
            except:pass