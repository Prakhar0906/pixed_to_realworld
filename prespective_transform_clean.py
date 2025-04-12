import cv2 as cv
import numpy as np

imgpoints = []
font = cv.FONT_HERSHEY_SIMPLEX

def print_coords(event,x,y,falgs,params):
    if event == cv.EVENT_LBUTTONDOWN:
        coords.append((x,y))


cap = cv.VideoCapture(1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

ret, corners = cv.findChessboardCorners(gray, (6,9), None)

corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners)
corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners2)
corners2 = corners2.astype(np.int64).tolist()

print(corners2[0][0],corners2[5][0],corners2[48][0],corners2[53][0])

tl = corners2[0][0]
bl = corners2[48][0]
tr = corners2[5][0]
br = corners2[53][0]
t_x = 500
t_y = 780


cv.namedWindow('image')
cv.setMouseCallback('image',print_coords)

coords = []

while True:
    true,frame = cap.read()
    

    if not true:
        break
        
    pts1 = np.float32([tl,bl,tr,br])

    pts2 = np.float32([[0,0],[0,t_y],[t_x,0],[t_x,t_y ]])


    
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    tframe = cv.warpPerspective(frame,matrix,(t_x,t_y))
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    tframe = cv.filter2D(tframe, -1, kernel)
    
    cv.putText(tframe,"(0,0)",[0,0], font,1,(255,255,255),1,cv.LINE_AA)

    for i in coords:
        val = str(format(i[0]*0.024, '.2f')) + ',' + str(format(i[1]*0.025, '.2f'))
        cv.putText(tframe,val,i, font,0.5,(255,0,0),1,cv.LINE_AA)
        
 
    cv.imshow('image',tframe)
    
    if cv.waitKey(1) == ord('q'):
        break   


cap.release()
cv.destroyAllWindows()

