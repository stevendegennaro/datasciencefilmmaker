# Averging Video Frames to Enhance Features

Uses [OpenCV](https://pypi.org/project/opencv-python/) to import a movie file frame by frame and save those frames as a numpy array in a file to disk.

Uses that file as a memmap to create a running average of each pixel.

If the video is still, averaging each pixel should remove noise. The final still image created is then output to disk or displayed on screen.

sign_average.py outputs the final image to disk.

running_average.py plots the image on screen using pyplot as the average is calculated, then outputs the final result to disk.