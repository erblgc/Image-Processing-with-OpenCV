import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("life.jpg") # original image
imgray = cv2.imread("life.jpg",cv2.IMREAD_GRAYSCALE) # grayscale image
imgContour = img.copy() # copy of original image
imgaus = cv2.adaptiveThreshold(imgray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3) # morphological image
rows, cols,_ = img.shape # 1024 x 683

cropped = img[0:215, 0:250] # cropped image
resized = cv2.resize(img,(1500,683)) # resized to 1500 x 683

rotation = cv2.getRotationMatrix2D((rows/2, cols/2), 45, 1) # 45 degree rotation matrix
rotated = cv2.warpAffine(img, rotation, (cols, rows)) # rotated image

H_flip = cv2.flip(img, 1) # horizontal flip
V_flip = cv2.flip(img, 0) # vertical flip

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Color Space Conversion
hue = hsv[:,:,0] # hue channel

gaussianBlur = cv2.GaussianBlur(img, (25,25), 0) # blurred image

cannyEdge = cv2.Canny(img, 180, 180) # canny edge detection

contours, _ = cv2.findContours(cannyEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # contours detection
cv2.drawContours(imgContour, contours, -1, (0,255,0), 3) # draw contours

# histogram equalization
imgH = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equ = cv2.cvtColor(imgH, cv2.COLOR_GRAY2BGR)
e,q,u = cv2.split(equ)
equ = cv2.equalizeHist(e)

# thresholding
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, thresh1 = cv2.threshold(gray, 123, 252, cv2.THRESH_BINARY)

# erosion 
kernel = np.ones((2,2), np.uint8)
erosion = cv2.erode(imgaus, kernel, iterations=1)

# dilation
dilation = cv2.dilate(imgaus, kernel, iterations=1)

# orb detection
orb = cv2.ORB_create()
kp = orb.detect(imgray, None)
kp, des = orb.compute(imgray, kp)
img2 = cv2.drawKeypoints(imgray,kp , imgray,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.subplot(4, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.colorbar()

plt.subplot(4, 4, 2)
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')
plt.colorbar()

plt.subplot(4, 4, 3)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title('Resized Image')
plt.colorbar()

plt.subplot(4, 4, 4)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')
plt.colorbar()

plt.subplot(4, 4, 5)
plt.imshow(cv2.cvtColor(H_flip, cv2.COLOR_BGR2RGB))
plt.title('Horizontal Flip')
plt.colorbar()

plt.subplot(4, 4, 6)
plt.imshow(cv2.cvtColor(V_flip, cv2.COLOR_BGR2RGB))
plt.title('Vertical Flip')
plt.colorbar()

plt.subplot(4, 4, 7)
plt.imshow(hue)
plt.title('Hue')
plt.colorbar()

plt.subplot(4, 4, 8)
plt.imshow(cv2.cvtColor(gaussianBlur, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Blur')
plt.colorbar()

plt.subplot(4, 4, 9)
plt.imshow(cannyEdge, cmap='gray')
plt.title('Canny Edge Detection')
plt.colorbar()

plt.subplot(4, 4, 10)
plt.imshow(cv2.cvtColor(imgContour, cv2.COLOR_BGR2RGB))
plt.title('Contours')
plt.colorbar()

plt.subplot(4, 4, 11)
plt.imshow(cv2.cvtColor(equ, cv2.COLOR_BGR2RGB))
plt.title('Histogram Equalization')
plt.colorbar()

plt.subplot(4, 4, 12)
plt.imshow(thresh1, cmap='gray')
plt.title('Thresholding')
plt.colorbar()

plt.subplot(4, 4, 13)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.colorbar()

plt.subplot(4, 4, 14)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.colorbar()

plt.subplot(4, 4, 15)
plt.imshow(img2, cmap='gray')
plt.title('ORB Detection')
plt.colorbar()

plt.subplots_adjust(left=0.005, bottom=0.11, right=0.957, top=0.917, wspace=0.171, hspace=0.267)
plt.show()

plt.xticks(np.arange(0, cols, 100))
plt.yticks(np.arange(0, rows, 100))
cv2.waitKey(0)