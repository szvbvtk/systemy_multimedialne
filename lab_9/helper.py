import cv2

import matplotlib.pyplot as plt

file = "VIDEOS/clip_3.mp4"

# Open the video file
cap = cv2.VideoCapture(file)

# Read the first frame
ret, frame = cap.read()
# Read 65 frames
for _ in range(349):
    ret, frame = cap.read()

# Convert the frame from BGR to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame in a matplotlib plot
plt.imshow(frame_rgb)
plt.axis("on")
plt.show()

# Release the video capture object
cap.release()
