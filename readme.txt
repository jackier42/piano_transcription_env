## README FOR PIANO TRANSCRIPTION PROJECT ##

Annabelle Ritchie, 2020

Run cardboard_piano.py using Python 3.7 and the libaries listed below. I've included the sample video file (curtains_closed_hand.mp4) I used in the zip as an example. The only requirements for video files for this program are:
- first frame needs to be an unobstructed keyboard
- keyboard needs to be in view at all times

LIBRARY VERSIONS USED IN PROJECT:
numpy==1.18.2
opencv-contrib-python==4.2.0.34
opencv-python==4.2.0.32


CODE STRUCTURE
setup(): run at the start of the program
main_loop(): this is the main loop where everything else is called - work through this function to see the order of all other functions, roughly:
> canny_edge_detection: canny edge algorithm
> background_subtraction: isolates the hand by colour
> houghP: finds hough lines
> erode_dilate: morphological operations, opening and closing
> contoured: contouring, convex hull, convexity defects
> draw_final_img: puts all the information together for the final frame