import numpy as np
from scipy import linalg
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

K = np.matrix('332.48603881 0 115.120247; 0 331.7923348 79.46300193; 0 0 1')
img1 = cv2.imread('first.jpg')
img2 = cv2.imread('second.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1=sift.detect(gray1,None)
kp2=sift.detect(gray2,None)

kp1,des1 = sift.compute(gray1,kp1)
kp2,des2 = sift.compute(gray2,kp2)
cv2.drawKeypoints(gray1,kp1,img1,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(gray2,kp2,img2,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print des2
print des1
cv2.imwrite('first2.jpg',img1)
cv2.imwrite('second2.jpg',img2)

bf = cv2.BFMatcher()

matches = bf.knnMatch(des1,des2,k=2)
good = []
pts1 = []
pts2 = []
for m,n in matches:
	if m.distance < 0.75*n.distance:
		good.append([m])
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)


pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

def drawlines(gray1,gray2,lines,pts1,pts2):
	r,c = gray1.shape
	img1 = cv2.cvtColor(gray1,cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(gray2,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0,-r[2]/r[1]])
		x1,y1 = map(int, [c,-(r[2]+r[0]*c)/r[1]])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(gray1,gray2,lines1,pts1,pts2)

lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
lines2 = lines2.reshape(-1,3)

img3,img4 = drawlines(gray2,gray1,lines2,pts2,pts1)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()

img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.imwrite('Sift.jpg',img3)
plt.imshow(img3),plt.show()
print des1
print "The number of matches founds is %0.0f" % len(good)
"""E=TRANS+INVERSE(K)*F*K"""
Kpr = np.transpose(K)
E = Kpr*F*K
print "The Essential matrix is: \n"
print E
print "The Fundamental matrix is: \n"
print F
""" assuming the first camera extrinsic parameter has the following form = [1 0 0 0;0 1 0 0; 0 0 1 0]"""

points, R, t, mask = cv2.recoverPose(E,pts1,pts2)
print "The rotational matrix is: "
print R
print "The translation vector is: "
print t
points1 = np.transpose(pts1)
R2=np.transpose(R)
t2=np.dot(R2,t)
print t
print t2
print R
print R2

points2 = np.transpose(pts2)
camMatrix2 = np.concatenate((R2,-t2),axis=1)
camMatrix1 = np.concatenate((R,t),axis=1)
p1=np.dot(K,camMatrix1)
p2=np.dot(K,camMatrix2)
print p1
print p2
X=[]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(0,92):
	X.append(cv2.triangulatePoints(p1,p2,points1[:,[i]],points2[:,[i]]))
	X[i] /= X[i][3]
	xs = X[i][0]
	ys = X[i][1]
	zs = X[i][2]
	ax.scatter(xs,ys,zs,zdir='z',c=i)
X /=X[3]
print X
print len(X)
print X[:,1]
print X[:,2]
print X.shape
with open("Results.txt","w") as text_file:
	text_file.write("{}".format(X))
	text_file.write("{}".format(p1))
	text_file.write("{}".format(p1))
plt.show()
