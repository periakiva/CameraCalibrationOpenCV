import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
objp=np.zeros((8*6,3),np.float32)
objp[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)

imgpoints = []
objpoints = []
images = glob.glob('/path/to/images/*.jpg')
print "hello"
print (images)
for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret = False
	ret, corners=cv2.findChessboardCorners(gray,(8,6))
	print (ret)
	print (corners)
	if ret == True:
		cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners)
		objpoints.append(objp)
		print "hello"
		cv2.drawChessboardCorners(img,(8,6),corners,ret)
		cv2.imshow('img',img)
		cv2.imwrite('imageCalibration.JPG',img)
		cv2.waitKey(500)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)


img = cv2.imread('24.jpg')
h,w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)

x,y,w,h=roi
dst = dst[y:y+h,x:x+w]
cv2.imwrite('calibresult.jpg',dst)

