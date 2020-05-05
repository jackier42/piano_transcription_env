# hough_line.py

import cv2
import numpy as np
import operator as op
from copy import deepcopy

## Constants ##
# CAMERA NUMBER CONSTANT
# CHANGE THIS IF NO CAMERA OR INCORRECT CAMREA IS PICKED UP
CAM_NUM = 1

# Keys
ESC = 27

# Thresholds
MIN_CONTOUR_AREA = 10000
MAX_DIST = 34
ANGLE_MAX = 60
VEL_THRESH = 50

# Colours
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)

# Wait time for loop
WAIT = 100

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def dist(p1, p2):
    """Calculates the Euclidean distance between two 2D points"""
    dx = p1[0][0] - p2[0][0]
    dy = p1[0][1] - p2[0][1]
    mag = np.sqrt(dx**2 + dy**2)
    return mag

def group(data, radius, dist_func=dist):
    """Clusters data that fall within radius of each by measure of dist_func"""
    d = data.copy()
    
    d = [x for x in d]
    
    #d.sort(key=lambda x: dist([[0, 0]], x))
    
    diff = [dist_func(p1, p2) for p1, p2 in zip(*[iter(d)] * 2)]
    
    m = [[d[0]]]
    
    for x in d[1:]:
        if dist(x, m[-1][0]) < radius:
            m[-1].append(x)
        else:
            m.append([x])
        
    return m

def avg_of_groups(point_groups):
    """Given a list of groups of 2D points, returns a list with the averages 
    of each group in the same order"""
    point_avgs = []
    
    for group in point_groups:
        point_avg = [[0, 0]]
        
        for point in group:
            point_avg[0][0] += point[0][0]
            point_avg[0][1] += point[0][1]
            
        point_avg[0][0] = int(round(point_avg[0][0]/len(group)))
        point_avg[0][1] = int(round(point_avg[0][1]/len(group)))
        
        point_avgs += [point_avg]
    
    return point_avgs

def index_of_closest(data, points, dist_func=dist):
    """Generates an index of the closest values in data to values in points by
    measure of dist_func"""
    d = deepcopy(data)
    p = deepcopy(points)
    
    ioc = []
    
    for point in points:
        dist_curr = 99999
        indx = -1
        for i, dpoint in enumerate(data):
            disti = dist_func(point, dpoint) 
            if disti < dist_curr:
                dist_curr = disti 
                indx = i
        ioc += [indx]
        
    return ioc

def draw_hough_line(rho, theta, img, colour):
    """Draws a Hough line on image img with colour colour given the Hough line
    paramers rho and theta"""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), colour, 2)    

def sep_hor_and_ver(lines, thresh=np.pi/4):
    """Seperates lines parameterised in Hough space by which lines are horizontal
    and which lines are vertical"""
    hor_lines = []
    ver_lines = []            

    if lines is None:
        return None, None

    for line in lines:
        #print(line)
        for rho, theta in line:
            
            h_point = (rho, theta)
            
            theta_ver_check = (theta + np.pi/2) % (2*np.pi) # Rotate 90 deg
            theta_ver_check = abs(theta_ver_check - np.pi) # Combine top and bottom angle ranges
            theta_ver_check = abs(theta_ver_check  - np.pi/2) # Combone left and right angle ranges
            
            theta_hor_check = abs(theta - np.pi) # Combine top and bottom angle ranges
            theta_hor_check = abs(theta_hor_check  - np.pi/2) # Combone left and right angle ranges
            
            if theta_ver_check < thresh:
                if not h_point in ver_lines:
                    ver_lines += [h_point]
                #draw_hough_line(rho, theta, img, (0, 0, 255))
            elif theta_hor_check < thresh:
                if not h_point in hor_lines:
                    hor_lines += [h_point]
                #draw_hough_line(rho, theta, img, (255, 0, 0)) 
                
    return hor_lines, ver_lines

def main():
    """Main function that does all image processing"""
    # Set camera
    cap = cv2.VideoCapture("videos/curtains_closed_hand.mp4")#CAM_NUM)
    
    # Read first image from camera
    ret, img_original = cap.read()
    
    #cv2.imwrite('img_org.png', img_original)
    
    height, width, _ = img_original.shape;
    
    print(height, width)
    
    rangeL = np.array([0, 0, 0])
    rangeH = np.array([180, 255, 80])
    #rangeL = np.array([0, 0, 180])
    #rangeH = np.array([180, 50, 255])    
    
    cv2.namedWindow('HLT')
    cv2.createTrackbar('CannyThreshold1', 'HLT', 0, 1200, nothing)
    cv2.createTrackbar('CannyThreshold2', 'HLT', 0, 1200, nothing)    
    cv2.createTrackbar("HoughThreshold", 'HLT', 0, 200, nothing)
    cv2.createTrackbar("Angle Res", "HLT", 10, 360, nothing)
    
    cv2.setTrackbarPos('CannyThreshold1', 'HLT', 200)
    cv2.setTrackbarPos('CannyThreshold2', 'HLT', 250)
    cv2.setTrackbarPos("HoughThreshold", 'HLT', 70)  
    cv2.setTrackbarPos("Angle Res", "HLT", 180)
    
    cannyThreshold1 = cv2.getTrackbarPos('CannyThreshold1', 'HLT')
    cannyThreshold2 = cv2.getTrackbarPos('CannyThreshold2', 'HLT')
    houghThreshold = cv2.getTrackbarPos('HoughThreshold', 'HLT')
    angleRes = cv2.getTrackbarPos("Angle Res", "HLT")
    
    # Create a new copy of the original image for drawing on later.
    img = img_original.copy()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Find edges of piano
    thresh = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), rangeL, rangeH)
    kernel = np.ones((3, 3), np.uint8)
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)  
    gray = close
    edges = cv2.Canny(gray, cannyThreshold1, cannyThreshold2)
    lines = cv2.HoughLines(edges, 1, np.pi/angleRes, houghThreshold)
    
    cv2.imshow('gray', gray)

    M = np.empty([1,1])
    
    # For each line that was detected, draw it on the img.     
    if lines is not None:
        # Seperate veritcal and horizontal lines and sort the lines by rho
        hor_lines, ver_lines = sep_hor_and_ver(lines)
        ver_lines.sort(key = lambda r: abs(op.itemgetter(0)(r)))
        hor_lines.sort(key = lambda r: abs(op.itemgetter(0)(r)))
        
        # Draw
        if ver_lines is not None:
            for rho, theta in ver_lines:
                draw_hough_line(rho, theta, img, RED)    
        if hor_lines is not None:
            for rho, theta in hor_lines:
                draw_hough_line(rho, theta, img, BLUE)          
        
        points = []           
        if not ver_lines == [] and not hor_lines == []:
            # Calculate four intersections points of outer-most edges
            for rho1, theta1 in [ver_lines[0], ver_lines[-1]]:
                for rho2, theta2 in [hor_lines[0], hor_lines[-1]]:
                    gam1 = rho1 / (np.sin(theta1) + 0.00001)
                    chi1 = 1 / (np.tan(theta1) + 0.00001)
                    gam2 = rho2 / (np.sin(theta2) + 0.00001)
                    chi2 = 1 / (np.tan(theta2) + 0.00001)
                    
                    x = int(round((gam2 - gam1) / (chi2 - chi1)))
                    y = int(round(gam1 - x*chi1))
                    
                    points += [[x, y]]
                    
                    # Draw the points
                    cv2.circle(img, (x, y), 2, (0, 255, 0), 2)
                  
            pts1 = np.float32(points)
            pts2 = np.float32([[width, 300], [width, 0], [0, 300], [0, 0]])
            
            #print(points)
            
            #Calculate Homography transformatino
            M = cv2.getPerspectiveTransform(pts1, pts2)
            print(M)
            frame0 = cv2.warpPerspective(img_original, M, (width, 300))
            
            #cv2.imwrite('image_find.png', img)
            combined = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)
            cv2.imshow('HLT', combined)
            
            # Find lines on transformed view of piano
            rangeL_f = np.array([0, 0, 0])
            rangeH_f = np.array([180, 255, 80])            
            grey_f = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            grey_f = cv2.blur(grey_f, (2, 2))
            thresh_f = cv2.inRange(cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV), rangeL_f, rangeH_f)
            kernel_f = np.ones((2, 2), np.uint8)
            #dilate_f = cv2.morphologyEx(thresh_f, cv2.MORPH_DILATE, kernel_f, iterations=4)  
            gray_f = thresh_f
            edges_f = cv2.Canny(grey_f, 280, 280)
            lines_f = cv2.HoughLines(edges_f, 1, np.pi/angleRes, 110)
            
            # Find verticle lines (edges of keys) and sort by rho
            lines_f = avg_of_groups(group(lines_f, 50, lambda p1, p2: abs(abs(p1[0]) - abs(p2[0]))))
            hor_lines_f, key_seg_lines = sep_hor_and_ver(lines_f, np.pi/32)
            key_seg_lines.sort(key = lambda r: abs(op.itemgetter(0)(r)))
            
            #print(key_seg_lines)
            
            # Draw
            if key_seg_lines is not None:
                for rho, theta in key_seg_lines:
                    draw_hough_line(rho, theta, frame0, RED)
            
            #cv2.imshow('edges_f', edges_f)    
            cv2.imshow('frame0', frame0) 
            #cv2.imwrite('frame0.png', frame0)            
    
    # Control loop        
    prev_hands = []
    while True:
        # Transform frame to normalised view
        frame1 = cv2.warpPerspective(img_original, M, (width, 300))
        
        # Isolate hands
        skinRangeL1 = np.array([8, 0, 0])
        skinRangeH1 = np.array([172, 75, 255])
        skinRangeL2 = np.array([0, 0, 0])
        skinRangeH2 = np.array([180, 255, 75])
        frame1_blur = cv2.blur(frame1, (9, 9))
        skin1 = cv2.inRange(cv2.cvtColor(frame1_blur, cv2.COLOR_BGR2HSV), skinRangeL1, skinRangeH1)
        skin2 = cv2.inRange(cv2.cvtColor(frame1_blur, cv2.COLOR_BGR2HSV), skinRangeL2, skinRangeH2)
        skin = cv2.bitwise_or(skin1, skin2)
        skin = cv2.bitwise_not(skin) 
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, (5, 5), iterations=20)
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, (7, 7), iterations=20)
        
        # Find hand contours
        im2, contours, hierarchy = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find one-to-two bigest contours with a large enough area, find their
        # convex hull, refined the hull to only have one corner in a local
        # neighbourhood, and find convexity defects
        hand_contours = []
        hand_convexityDefects = []
        if (len(contours) > 0):
            contours.sort(key=cv2.contourArea, reverse=True)
            
            contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
            
            convex1 = []
            convex2 = []
            if (len(contours) > 0):
                convex1 = cv2.convexHull(contours[0])
                convex1_g = np.array(avg_of_groups(group(convex1, MAX_DIST)))
                convex1d = cv2.convexHull(contours[0], returnPoints=False)  
                convex1_gd = np.array(index_of_closest(contours[0], convex1_g))
                defects1 = cv2.convexityDefects(contours[0], convex1_gd)
                
                #cv2.drawContours(frame1, [convex1_g], -1, GREEN, 2)
                #cv2.drawContours(frame1, [contours[0]], -1, BLUE, 2)                
                hand_contours += [convex1]
                hand_convexityDefects += [defects1]
                
                if (len(contours) > 1):
                    convex2 = cv2.convexHull(contours[1])
                    convex2_g = np.array(avg_of_groups(group(convex2, MAX_DIST)))
                    convex2d = cv2.convexHull(contours[1], returnPoints=False)
                    convex2_gd = np.array(index_of_closest(contours[1], convex2_g))
                    defects2 = cv2.convexityDefects(contours[1], convex2_gd)
                    
                    #cv2.drawContours(frame1, [convex2_g], -1, GREEN, 2)
                    #cv2.drawContours(frame1, [contours[1]], -1, BLUE, 2)                    
                    hand_contours += [convex2]
                    hand_convexityDefects += [defects2]
            
            #print(p)
            pass
        
        # Calculate the positions of the fingertips using convex hull and
        # convexity defects
        if hand_convexityDefects is not None:
            hands = []
            for i, convexityDefects in enumerate(hand_convexityDefects):
                #print(convexityDefects)
                contour = contours[i]
                fingertips = []
                for j, defects in enumerate(convexityDefects):
                    #print(line)

                    s = defects[0][0]
                    e = defects[0][1]
                    f = defects[0][2]
                    d = defects[0][3]
                    
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    a = dist([start], [end])
                    b = dist([far], [start])
                    c = dist([far], [end])
                    
                    angleDeg = np.arccos((b**2 + c**2 - a**2)/(2*b*c))*(180/np.pi)
                    
                    if angleDeg < ANGLE_MAX:
                        if not start in fingertips:
                            fingertips += [start]
                        if not end in fingertips:
                            fingertips += [end]
                                          
                        #cv2.circle(frame1, start, 5, WHITE, -1)
                        #cv2.circle(frame1, end, 5, WHITE, -1)
                        #cv2.circle(frame1, far, 5, RED, -1)
                        
                hands += [fingertips]
           
        # Calculate y-velocity (y-difference) of each fingertip and use it to
        # determine when a finger is pressing
        pressed_fingers = []            
        for i, hand in enumerate(hands):
            try:
                prev_hand = prev_hands[i]
            except:
                prev_hand = hand
            
            for j, fingertip in enumerate(hand):
                try:
                    prev_fingertip = prev_hand[j]
                except:
                    prev_fingertip = fingertip   
                    
                ydiff = prev_fingertip[1] - fingertip[1]
                
                if ydiff > VEL_THRESH:
                    colour = RED
                    pressed_fingers += [fingertip]
                else:
                    colour = CYAN
                
                # Draw fingers tips (RED for press, CYAN otherwise)
                cv2.circle(frame1, fingertip, 8, colour, 2)
        
        # Highlight the pressed key(s)        
        for point in pressed_fingers:
            key_seg_lines.sort(key=lambda x: abs(x[0] - point[0]))
            
            draw_hough_line(key_seg_lines[0][0], key_seg_lines[0][1], frame1, MAGENTA)
            draw_hough_line(key_seg_lines[1][0], key_seg_lines[1][1], frame1, MAGENTA)
            #cv2.imwrite('frame_pressed.png', frame1)
        
        # Save previous state of hands
        prev_hands = deepcopy(hands)
        
        cv2.imshow('skin', skin)
        cv2.imshow('frame1', frame1)
        
        key = cv2.waitKey(WAIT)
        
        # Save images on space press
        if key & 0xFF == ord(' '):
            cv2.imwrite('skin.png', skin)
            cv2.imwrite('frame.png', frame1)
            print("Images written!")
            
        # Press ESC to exit
        if key & 0xFF == ESC:
            cap.release()
            cv2.destroyAllWindows()

            print(ver_lines)
            print(hor_lines)
            
            print(x, y)

            break
        
        #img_original = cv2.imread('img.png')
        ret, img_original = cap.read()      

if __name__ == "__main__":
    main()
    #houghNormal()
