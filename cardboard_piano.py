import numpy as np
import cv2

CAMERA_INDEX = 0

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

def erode_dilate(img):
    kernel = np.ones((10, 10), np.uint8) 
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing

def contoured(img, mask):
    image_copy = img.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_copy, contours, -1, (52, 235, 131))
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
    colour_deviation = cv2.getTrackbarPos('Colour Deviation','Background Subtraction') #90
    lower_skin = np.array([190 - colour_deviation, 140 - colour_deviation, 140 - colour_deviation])
    upper_skin = np.array([190 + colour_deviation, 140 + colour_deviation, 140 + colour_deviation])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    return mask


def setup():
    cap = cv2.VideoCapture("videos/useful_piano_clip.mp4")
    #cap = cv2.imread('images/resized_3.jpg')
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
