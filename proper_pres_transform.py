import cv2 as cv
import numpy as np




imgpoints = []
font = cv.FONT_HERSHEY_SIMPLEX

def nothing(x):
    pass

def print_coords(event,x,y,falgs,params):
    if event == cv.EVENT_LBUTTONDOWN:
        #200,490
        # 100 pixel is 2.4cm then 1 pixel is 0.024 cm
        print(x*0.024,y*0.025)
        

cap = cap = cv.VideoCapture(1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, frame = cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

ret, corners = cv.findChessboardCorners(gray, (6,9), None)
print(ret,corners)


corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners)
corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners2)
corners2 = corners2.astype(np.int64).tolist()

'''
for i in range(len(corners2)):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(frame,str(i),corners2[i][0], font,1,(255,255,255),1,cv.LINE_AA)
    cv.circle(frame, corners2[i][0], 3, (0,0,255),-1)
'''

print(corners2[0][0],corners2[5][0],corners2[48][0],corners2[53][0])

tl = corners2[0][0]
bl = corners2[48][0]
tr = corners2[5][0]
br = corners2[53][0]
t_x = 500
t_y = 780

print("The difference in pixel dx is:",corners2[0][0][0] - corners2[1][0][0])
print("The difference in pixel dy is:",corners2[0][0][1] - corners2[4][0][1])


imgpoints.clear()

cv.namedWindow('frame')
cv.setMouseCallback('frame',print_coords)
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
    
 
    cv.imshow('frame',tframe)
    
    
    #cv.imshow('frame',tframe)
    #cv.imshow('res',mask)
    
    if cv.waitKey(1) == ord('q'):
        break   


cap.release()
cv.destroyAllWindows()



'''

#tl = [442, 112]
#bl = [414,540]
#tr = [738, 118]
#br = [847, 524]

'''
'''
pts1 = np.float32([tl,bl,tr,br])
pts2 = np.float32([[0,0],[0,t_y],[t_x,0],[t_x,t_y ]])
cv.imshow("lol",transform_frame)
gray = cv.cvtColor(transform_frame, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (4,6), None)
print(ret,corners)
#corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners)
corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
imgpoints.append(corners2)
corners2 = corners2.astype(np.int64).tolist()
for i in range(len(corners2)):
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(transform_frame,str(i),corners2[i][0], font,1,(255,255,255),1,cv.LINE_AA)
    cv.circle(transform_frame, corners2[i][0], 3, (0,0,255),-1)

print("The difference in pixel dx is:",corners2[0][0][0] - corners2[1][0][0])
print("The difference in pixel dy is:",corners2[0][0][1] - corners2[4][0][1])

#print(corners2)
#cv.drawChessboardCorners(transform_frame, (4,7), corners2, ret)


cv.imshow("lol",transform_frame)

#cv.imshow("frame",frame)

'''


