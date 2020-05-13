import cv2 as cv
import numpy as np

shw = cv.imshow

frameWidth,widthImg = 640,640
frameHeight,heightImg = 480,480
print("Import Suceesfull")
cap = cv.VideoCapture(2)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,150)

def preprocesing(img):
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDialation = cv.dilate(imgCanny,kernel, iterations=2)
    imgThres = cv.erode(imgDialation,kernel,iterations =1)
    return imgThres
def getContours(img):
    maxarea = 0
    biggest = np.array([])
    contours, heirarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
    
        area = cv.contourArea(cnt)
        
        if area>5000:
            # cv.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri = cv.arcLength(cnt,True)
            
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            if area>maxarea and len(approx) == 4 :
                
                biggest = approx
                maxarea = area
            # x,y,w,h = cv.boundingRect(approx)
            # cv.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            # shw("Contour",imgContour)
            # #getPerspective(x,y,w,h,imgContour)
    cv.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest

def reorder(myPoints):
    myPoints = myPoints.reshape(4,2)
    newPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def getWrap(img,biggest):
    
    imgOutput = img.copy()
    if biggest.shape[0]==4:
        biggest =reorder(biggest)
        points = np.float32(biggest)
        refrence = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv.getPerspectiveTransform(points,refrence)
        imgOutput = cv.warpPerspective(img,matrix,(widthImg,heightImg))
        #shw("Output",imgOutput)
    return imgOutput

while True:
    _, img = cap.read()
    img = cv.resize(img,(frameWidth,frameHeight))
    imgThres = preprocesing(img)
    imgContour = img.copy()
    
    #print(success)
    
    biggest = getContours(imgThres)
    imgWarped = getWrap(img,biggest)
    print(biggest.shape[0])
    stack = np.hstack([imgContour,imgWarped])
    shw("Feed",stack)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

