import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from IPython.display import YouTubeVideo, display, HTML
from base64 import b64encode



# Download assets (commented out - video already in folder)
# def download_and_unzip(url, save_path):
#     print(f"Downloading and extracting assests....", end="")

#     # Downloading zip file using urllib package.
#     urlretrieve(url, save_path)

#     try:
#         # Extracting zip file using the zipfile package.
#         with ZipFile(save_path) as z:
#             # Extract ZIP file contents in the same directory.
#             z.extractall(os.path.split(save_path)[0])

#         print("Done")

#     except Exception as e:
#         print("\nInvalid file.", e)

# URL = r"https://www.dropbox.com/s/ld535c8e0vueq6x/opencv_bootcamp_assets_NB11.zip?dl=1"

# asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_NB11.zip")

# Download if assest ZIP does not exists. 
# if not os.path.exists(asset_zip_path):
#     download_and_unzip(URL, asset_zip_path)  

# Load and display video from downloaded assets
video_input_file_name = "dance.mp4"  # Adjust filename based on what's in the zip

# Draw the bounding box
def drawRectangle(frame, bbox) :
    p1 = (int(bbox[0]), int (bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int (bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

# Display bounding box on each frane
def displayRectangle(frame, bbox) :
    plt.figure(figsize = (20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy); plt.axis('off')

# Attach text
def drawText(frame, txt, location, color = (50, 170, 50)) :
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)


# Define Bounding box
bbox = (93, 389, 92, 200)


if len(sys.argv) > 1 and sys.argv[1] == '1' :
    # Read video to get first frame
    video = cv2.VideoCapture(video_input_file_name)
    ok, frame = video.read()
    
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Show first frame without bounding box
    cv2.imshow('First Frame - Original', frame)
    
    # Create a copy and draw bounding box on it
    frame_with_bbox = frame.copy()
    drawRectangle(frame_with_bbox, bbox)
    
    # Show frame with bounding box
    cv2.imshow('First Frame - With Bounding Box', frame_with_bbox)
    
    # Wait for 'q' key press to quit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    video.release()
    
else :
    # Create the tracker instance
    # Set up tracker
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "CSRT",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE"
    ]

    # Change the index to change the tracker type
    tracker_type = tracker_types[0]
    
    # Set GOTURN model paths if using GOTURN
    if tracker_type == "GOTURN":
        goturn_model = "goturn.caffemodel"
        goturn_proto = "goturn.prototxt"

    if tracker_type == "BOOSTING" :
        tracker = cv2.legacy.TrackerBoosting.create()
    elif tracker_type == "MIL" :
        tracker = cv2.legacy.TrackerMIL.create()
    elif tracker_type == "KCF" :
        tracker = cv2.legacy.TrackerKCF.create()
    elif tracker_type == "CSRT" :
        tracker = cv2.legacy.TrackerCSRT.create()
    elif tracker_type == "TLD" :
        tracker = cv2.legacy.TrackerTLD.create()
    elif tracker_type == "MEDIANFLOW" :
        tracker = cv2.legacy.TrackerMedianFlow.create()
    elif tracker_type == "GOTURN" :
        params = cv2.TrackerGOTURN_Params()
        params.modelTxt = goturn_proto
        params.modelBin = goturn_model
        tracker = cv2.TrackerGOTURN_create(params)
    else :
        tracker = cv2.legacy.TrackerMOSSE.create()


    # Read input video and setup output video
    # Read video
    video = cv2.VideoCapture(video_input_file_name)
    ok, frame = video.read()

    # Exit if video not opened
    if not video.isOpened() :
        print("Could not open video")
        sys.exit()
    else :
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_output_file_name = 'dance-' + tracker_type + '.mp4'
    video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    # Define bounding box
    displayRectangle(frame, bbox)

    # Initialize tracker
    ok = tracker.init(frame, bbox)


    # Read frame and track object
    while True :
        ok, frame = video.read()
        if not ok :
            break

    # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate fps
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok :
            drawRectangle(frame, bbox)
        else :
            drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))
    
        # Display info
        drawText(frame, tracker_type + "Tracker", (80, 60))
        drawText(frame, "FPS : " + str(int(fps)), (80, 100))

        # Write frame to video
        video_out.write(frame)

    video.release()
    video_out.release()


