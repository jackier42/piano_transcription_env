import numpy as np
import cv2

CAMERA_INDEX = 0
ANGLE_MAX = 60

# Colours
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


def nothing(x):
    # callback for createTrackbar
    pass

def canny_edge_detection(img):
    threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Hough Line Transform') # about 60
    threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Hough Line Transform') # about 300
    edges = cv2.Canny(img, threshold1, threshold2)
    return edges

def houghP(img, edges):
    img = cv2.GaussianBlur(img, (9,9), 0)
    #gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    minLineLength = cv2.getTrackbarPos('Min Line Length', 'Hough Line Transform')
    maxLineGap = cv2.getTrackbarPos('Max Line Gap', 'Hough Line Transform')
    
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLinesP(edges, 1, np.pi/720, 50, minLineLength=minLineLength, maxLineGap=maxLineGap)

    # Create a new copy of the original image for drawing on later.
    img_copy = img.copy()
    
    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

    # Create a combined image of Hough Line Transform result and the Canny Line Detector result.
    combined = np.concatenate((img_copy, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), axis=1)
    return combined

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
    #print(defects)
    if defects is not None:
            hands = []
            for i, convexityDefects in enumerate(defects):
                if i < len(contours):
                    #print(convexityDefects)
                    contour = contours[i]
                    #print(contour)
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
                        #print("angledeg " + str(angleDeg))
                        if angleDeg < ANGLE_MAX:
                            if not start in fingertips:
                                fingertips += [start]
                            if not end in fingertips:
                                fingertips += [end]
                                              
                            cv2.circle(frame, start, 5, WHITE, -1)
                            cv2.circle(frame, end, 5, WHITE, -1)
                            cv2.circle(frame, far, 5, RED, -1)
                            
                hands += [fingertips]

def contoured(img, mask):
    image_copy = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    defects_on_hand = []
    contours_in_hand = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        #print(area)
        if (area > 20000):#50000):
            hull_drawing = cv2.convexHull(contours[i])
            cv2.drawContours(image_copy, [hull_drawing], -1, (52, 235, 131))

            hull_defects = cv2.convexHull(contours[i], returnPoints=False)
            res = hull_defects
            defects = cv2.convexityDefects(contours[i], hull_defects)
            defects_on_hand.append(defects)
            contours_in_hand.append(contours[i])
        cv2.drawContours(image_copy, contours, -1, (52, 200, 300))
    find_fingers_using_defects(contours_in_hand, defects_on_hand, image_copy)
    return image_copy

def background_subtraction(img, fgbg):
##    foreground_mask = fgbg.apply(img)
##    ret, thresholded = cv2.threshold(foreground_mask,250,255,cv2.THRESH_BINARY)
##    return thresholded
    #rangeL = np.array([0, 0, 0])
    #rangeH = np.array([359, 100, 100])
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #th, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_TRUNC)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colour_deviation = 38 #cv2.getTrackbarPos('Colour Deviation','Background Subtraction') #90
    lower_skin = np.array([200 - colour_deviation, 69 - colour_deviation, 180 - colour_deviation])
    #lower_skin = np.array([170 - colour_deviation, 120 - colour_deviation, 125 - colour_deviation])
    upper_skin = np.array([200 + colour_deviation, 69 + colour_deviation, 180 + colour_deviation])
    #upper_skin = np.array([170 + colour_deviation, 120 + colour_deviation, 125 + colour_deviation])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    return mask


def setup():
    cap = cv2.VideoCapture("videos/curtains_closed_hand.mp4")
    #cap = cv2.imread('images/resized_3.jpg')
    #cap = cv2.VideoCapture(0)
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



def main_loop(cap, fgbg):
    ret, original_img = cap.read()
    
    #original_img = cv2.imread("images/resized_4.jpg")
    #cv2.imshow('frame', original_img)

    draw_img = original_img.copy()
    blur = cv2.GaussianBlur(original_img, (9,9), 0)
    edges = canny_edge_detection(blur)
    #cv2.imshow('Canny', edges)
    background_subtracted = background_subtraction(original_img, fgbg)
    
    #cv2.imshow('HoughP', houghP(draw_img, edges))
    #cv2.resize(background_subtracted, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Background Subtraction', background_subtracted)
    
    morphed_img = erode_dilate(background_subtracted)
    cv2.imshow('Morphological Transformations', morphed_img)

    contouring = contoured(original_img, morphed_img)
    cv2.imshow('Contouring', contouring)

    

if __name__ == "__main__":
    
    (cap, fgbg) = setup()
    
    while not (cv2.waitKey(30) & 0xFF == ord('q')):
        main_loop(cap, fgbg)
    
    cap.release()
    cv2.destroyAllWindows()
