import numpy as np
import cv2
from copy import deepcopy

CAMERA_INDEX = 0
ANGLE_MAX = 60
VEL_THRESH = 10

# Colours
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)

prev_hand = []
prev_hands = []

firsttime = True
hough_lines = []
hor_lines = []
ver_lines = []

def nothing(x):
    # callback for createTrackbar
    pass

def sep_hor_and_ver(lines, thresh=np.pi/4):
    """Seperates lines parameterised in Hough space by which lines are horizontal
    and which lines are vertical"""
    hor_lines = []
    ver_lines = []            

    if lines is None:
        return None, None

    for line in lines:
        #print(line)
        line = line[0]
        x = line[2] - line[0]
        y = line[3] - line[1]
        m = y/x
        theta = np.arctan(m)
        #print(theta)

        theta_ver_check = (theta + np.pi/2) % (2*np.pi) # Rotate 90 deg
        theta_ver_check = abs(theta_ver_check - np.pi) # Combine top and bottom angle ranges
        theta_ver_check = abs(theta_ver_check  - np.pi/2) # Combone left and right angle ranges
        
        theta_hor_check = abs(theta - np.pi) # Combine top and bottom angle ranges
        theta_hor_check = abs(theta_hor_check  - np.pi/2) # Combone left and right angle ranges
        
        if theta_ver_check < thresh:
            ver_lines += [line]
            #draw_hough_line(rho, theta, img, (0, 0, 255))
        elif theta_hor_check < thresh:
            hor_lines += [line]
            #draw_hough_line(rho, theta, img, (255, 0, 0)) 
                
    return hor_lines, ver_lines

def canny_edge_detection(img):
    threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Hough Line Transform') # about 60
    threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Hough Line Transform') # about 300
    edges = cv2.Canny(img, threshold1, threshold2)
    return edges

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def houghP(img, edges):
    img = cv2.GaussianBlur(img, (9,9), 0)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Line Transform')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Line Transform')
    
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLinesP(edges, 1, np.pi/720, 50, minLineLength=minLineLength, maxLineGap=maxLineGap)

    hor_lines, ver_lines = sep_hor_and_ver(lines)
    
    # Create a new copy of the original image for drawing on later.
    img_copy = img.copy()
    
    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in hor_lines:
            x1,y1,x2,y2 = line
            cv2.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

        for line in ver_lines:
            x1,y1,x2,y2 = line
            cv2.line(img_copy,(x1,y1),(x2,y2),(255,0,0),2)

    for hline in hor_lines:
        for vline in ver_lines:
            hor_line = [[hline[0], hline[1]], [hline[2], hline[3]]]
            ver_line = [[vline[0], vline[1]], [vline[2], vline[3]]]
            intersection_point = line_intersection(hor_line, ver_line)
            cv2.circle(img_copy, intersection_point, 1, MAGENTA, 2)
            #find intersection points of horizontal and vertical lines

    # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
    combined = np.concatenate((img_copy, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)
    return (img_copy, hor_lines, ver_lines)

def dist(p1, p2):
    """Calculates the Euclidean distance between two 2D points"""
    dx = p1[0][0] - p2[0][0]
    dy = p1[0][1] - p2[0][1]
    mag = np.sqrt(dx**2 + dy**2)
    return mag

def erode_dilate(img):
    kernel = np.ones((10, 10), np.uint8) 
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def find_fingers_using_defects(contours, defects, frame):
    # find fingertips given contouring and convexity defects
    finger_positions = []
    if defects is not None:
            hands = []
            for i, convexityDefects in enumerate(defects):
                if i < len(contours):
                    contour = contours[i]
                    fingertips = []
                    for j, defects in enumerate(convexityDefects):
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
                                              
                            #cv2.circle(frame, start, 5, WHITE, -1)
                            #cv2.circle(frame, end, 5, WHITE, -1)
                            #cv2.circle(frame, far, 5, RED, -1)
                            #uncomment this to get all fingertips drawn on
                            
                            
                hands += [fingertips]

            # Calculate y-velocity (y-difference) of each fingertip and use it to
            # determine when a finger is pressing
            pressed_fingers = []
            global prev_hands

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
                    
                    if ydiff > VEL_THRESH or ydiff < -VEL_THRESH:
                        colour = RED
                        pressed_fingers += [fingertip]
                    else:
                        colour = CYAN
                    
                    # Draw fingers tips (RED for press, CYAN otherwise)
                    #cv2.circle(frame, fingertip, 8, colour, 2)
                    finger_positions.append((fingertip, colour))
                    
            prev_hands = deepcopy(hands)
            
    return finger_positions

def contoured(img, mask):
    image_copy = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects_on_hand = []
    contours_in_hand = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        # only take contour that is a certain area to avoid tiny contours getting through
        if (area > 20000):
            hull_drawing = cv2.convexHull(contours[i])
            cv2.drawContours(image_copy, [hull_drawing], -1, (52, 235, 131))

            hull_defects = cv2.convexHull(contours[i], returnPoints=False)
            res = hull_defects
            defects = cv2.convexityDefects(contours[i], hull_defects)
            defects_on_hand.append(defects)
            contours_in_hand.append(contours[i])
        cv2.drawContours(image_copy, contours, -1, (52, 200, 300))

    fingertips = find_fingers_using_defects(contours_in_hand, defects_on_hand, image_copy)
    
    return (image_copy, fingertips)

def background_subtraction(img, fgbg):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colour_deviation = 38  #cv2.getTrackbarPos('Colour Deviation','Background Subtraction') #90 #38 in backup

    # I chose 200, 69, 180 from a colour picker on my image. this would be a good thing to improve on!
    lower_skin = np.array([200 - colour_deviation, 69 - colour_deviation, 180 - colour_deviation])
    upper_skin = np.array([200 + colour_deviation, 69 + colour_deviation, 180 + colour_deviation])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    return mask


def setup():
    cap = cv2.VideoCapture("videos/curtains_closed_hand.mp4")
    cv2.namedWindow('Hough Line Transform')
    cv2.createTrackbar('Canny Threshold 1', 'Hough Line Transform', 60, 1200, nothing)
    cv2.createTrackbar('Canny Threshold 2', 'Hough Line Transform', 290, 1200, nothing)
    cv2.createTrackbar("Min Line Length", 'Hough Line Transform', 0, 100, nothing)
    cv2.createTrackbar("Max Line Gap", 'Hough Line Transform', 120, 500, nothing)

    cv2.namedWindow('Background Subtraction')
    cv2.createTrackbar('Colour Deviation', 'Background Subtraction', 0, 160, nothing)
    foreground_background = cv2.createBackgroundSubtractorMOG2()

    cv2.namedWindow('Morphological Transformations')
    
    return (cap, foreground_background)

def draw_final_img(img, fingertips, ver_lines, hor_lines):
    for pt in fingertips:
        cv2.circle(img, pt[0], 8, pt[1], 2)
        if (pt[1] == RED):
            ver_lines.sort(key=lambda x: abs(x[0] - pt[0][0]))

            # this essentially selects the closest vertical lines and highlights them.
            # identifying the actual key pressed (eg. note name) would be harder.
            
            line_1 = [(ver_lines[0][0], ver_lines[0][1]), (ver_lines[0][2], ver_lines[0][3])]
            line_2 = [(ver_lines[1][0], ver_lines[1][1]), (ver_lines[1][2], ver_lines[1][3])]

            cv2.line(img, line_1[0], line_1[1], MAGENTA, 2)
            cv2.line(img, line_2[0], line_2[1], MAGENTA, 2)     
    cv2.imshow('Final', img)

def main_loop(cap, fgbg):
    # uncomment all the cv2.imshow functions you want to see the step by step
    
    ret, original_img = cap.read()
    
    #original_img = cv2.imread("images/resized_4.jpg")
    #cv2.imshow('frame', original_img)

    draw_img = original_img.copy()
    blur = cv2.GaussianBlur(original_img, (9,9), 0)
    edges = canny_edge_detection(blur)
    cv2.imshow('Canny', edges)
    background_subtracted = background_subtraction(original_img, fgbg)

    global hough_lines
    global hor_lines
    global ver_lines
    global firsttime
    if (firsttime):
        # this sets up the keyboard image from the first frame read - so there shouldn't be a hand visible there.
        (hough_lines, hor_lines, ver_lines) = houghP(draw_img, edges)
        firsttime = False
    else:
        (hough_lines, sample, sample2) = houghP(draw_img, edges)
    cv2.imshow('HoughP', hough_lines)
    #cv2.resize(background_subtracted, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Background Subtraction', background_subtracted)
    
    morphed_img = erode_dilate(background_subtracted)
    cv2.imshow('Morphological Transformations', morphed_img)

    (contouring_img, fingertips) = contoured(original_img, morphed_img)
    cv2.imshow('Contouring', contouring_img)

    draw_final_img(contouring_img, fingertips, hor_lines, ver_lines)
     

if __name__ == "__main__":
    
    (cap, fgbg) = setup()
    
    while not (cv2.waitKey(30) & 0xFF == ord('q')):
        main_loop(cap, fgbg)
    
    cap.release()
    cv2.destroyAllWindows()
