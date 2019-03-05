import cv2
import numpy as np
import os


def process_img(frame):
	#Processing pictures to extract contours
	frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	hand_lower = np.array([0,41,69])
	hand_upper = np.array([30,152,220])
	mask = cv2.inRange(frame_hsv,hand_lower,hand_upper)
	ret,mask = cv2.threshold(mask,127,255,0)
	kernel = np.ones((7,7),np.uint8)
	mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.bilateralFilter(mask,5,75,75)
	return mask

def find_gra_centre(hull):
	#Obtain the center of gravity of the object to be detected
	x = 0
	y = 0
	for i in range(hull.shape[0]):
		x = x + hull[i][0][0]
		y = y + hull[i][0][1]
	gra_x = int(x/hull.shape[0])
	gra_y = int(y/hull.shape[0])
	gra_centre = (gra_x, gra_y)
	return gra_centre

def find_range_hull(hull):
	#Get the boundary of convex hull
	hull_array = np.array(hull)
	hull_array_t = hull_array[:,0,:]
	x_min ,y_min = np.min(hull_array_t,axis=0) 
	x_max ,y_max = np.max(hull_array_t,axis=0)
	img_range = [x_min,y_min, x_max,  y_max]
	return img_range
	
def draw_rectangle(img, img_range):
	#draw rectangle to extract object
	border_x =  int((img_range[2] - img_range[0])/6)
	border_y =  int((img_range[3] - img_range[1])/6)
	x_min = img_range[0] - border_x
	y_min = img_range[1] - border_y
	x_max = img_range[2] + border_x
	y_max = img_range[3] + border_y
	cv2.rectangle(img,(x_min, y_min),(x_max, y_max), (0,255,0))

def get_max_contour(contours):
	#obtain the maximum contour
	max=0
	ci = 0
	for i in range(len(contours)):                              #
		cnt = contours[i]                                   #      
		area = cv2.contourArea(cnt)                         #
		if area>max:                                        # 
			max = area                                      #
			ci = i                                          #
		cnt = contours[ci]
	return cnt
	
def draw_convexitydefects(cnt):
	#find the convexity defects and draw circle to sign them
	hull = cv2.convexHull(cnt, returnPoints= False)
	defects = cv2.convexityDefects(cnt,hull)
	global count
	for i in range(defects.shape[0]):      
		s,e,f,d = defects[i,0]                 
		#if d > 14000 and d<28000:                           
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		cv2.circle(frame,far,5,[0,0,255],-1)             
		count += 1                                      

def getROI(frame, img_range, size):
	#get the region of interest of a picture and resize its zise
	border_x =  int((img_range[2] - img_range[0])/6)
	border_y =  int((img_range[3] - img_range[1])/6)
	x_min = img_range[0] - border_x
	y_min = img_range[1] - border_y
	x_max = img_range[2] + border_x
	y_max = img_range[3] + border_y
	ROI = frame[y_min:y_max,x_min:x_max]  
	ROI = cv2.resize(ROI, (size[0], size[1]))
	return ROI


	
	
	
frame = cv2.imread('D:\\GitHub\\gesture_tracking\\picture\\pic_2.bmp') 
mask = process_img(frame)

_,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = get_max_contour(contours)

epsilon = 0.25*cv2.arcLength(cnt,True)

approx = cv2.approxPolyDP(cnt,epsilon,True)

hull = cv2.convexHull(cnt,returnPoints=True)

gra_centre = find_gra_centre(hull)
cv2.circle(frame, gra_centre,8,[0,0,255],-1)

img_range = find_range_hull(hull)
draw_rectangle(frame, img_range)

ROI = getROI(frame, img_range, (60,60))
cv2.imshow('1', ROI)

cv2.drawContours(frame,[cnt],0,(255,0,0),3)
cv2.drawContours(frame,[hull],0,(0,255,0),3)

count = 0
draw_convexitydefects(cnt)
print(count)

#font = cv2.FONT_HERSHEY_COMPLEX
#cv2.putText(frame,str(count+1),(100,100),font,1,(0,0,255),1)
cv2.imshow('frame',frame)

cv2.waitKey(0)
cv2.destroyAllWindows()