import cv2 as cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import time

cvNet = cv2.dnn.readNetFromTensorflow('exported/frozen_inference_graph.pb', 'exported/graph.pbtxt')
tesseract_conf = r'--oem 3 --psm 6'

headless = False
approved_number = ["B 3815 KXI", "E 6730 RC"]

# Callback function for trackbar
def on_change(self):
    pass

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

# Creates window and trackbar
if not headless: 
    cv2.namedWindow('orig_img')
    cv2.createTrackbar('Thresh', 'orig_img', 0, 255, on_change)


# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    start = time.time()
    
    # Capture frame-by-frame
    ret, img = cap.read()
    rows = img.shape[0]
    cols = img.shape[1]
    
    thresh = 0
    
    if ret == True:
        cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
        cvOut = cvNet.forward()
        crop_img = []

        for detection in cvOut[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3 and ((detection[5] - detection[3] <= .6) or (detection[6] - detection[4] <= .6)):
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                if left < 0: left = 0
                if top < 0: top = 0
                if right < 0: right = 0
                if bottom < 0: bottom = 0
                crop_img.append(img[int(top):int(bottom), int(left):int(right)])
                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)   

        if not headless: 
            cv2.imshow("orig_img",img)
            thresh = cv2.getTrackbarPos('Thresh', 'orig_img')
        else: thresh = 250
        
        for i in range(0, len(crop_img)):
            crop_img[i] = cv2.cvtColor(crop_img[i], cv2.COLOR_BGR2RGB)

            crop_img[i] = cv2.resize(crop_img[i], (680, 360))
            crop_img[i] = cv2.cvtColor(crop_img[i], cv2.COLOR_RGB2GRAY)
            #crop_img[i] = cv2.GaussianBlur(crop_img[i], (5,5), 0)
            #crop_img[i] = cv2.medianBlur(crop_img[i], 3)
            crop_img[i] = cv2.threshold(crop_img[i], thresh, 255, cv2.THRESH_BINARY)

            string = pytesseract.image_to_string(crop_img[i][1], lang="engeng", config=tesseract_conf)
            #print(string)
            for plateNumber in approved_number:
                if string.find(plateNumber) != -1:
                    print("Found", plateNumber)

            contours, _= cv2.findContours(crop_img[i][1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_poly = [None]*len(contours)
            boundRect = [None]*len(contours)

            for ix, c in enumerate(contours):
                contours_poly[ix] = cv2.approxPolyDP(c, 3, True)
                boundRect[ix] = cv2.boundingRect(contours_poly[ix])

            crop_img[i] = cv2.cvtColor(crop_img[i][1], cv2.COLOR_GRAY2BGR)
            letter = []
            
            if not headless: cv2.imshow("crop_img",crop_img[i])

        # Press Q on keyboard to  exit
        if not headless:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    # Break the loop
    else: 
        break
    
    print("FPS\t=\t", 1.0 / (time.time() - start))

# When everything done, release the video capture object
if not headless:
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()
