import cv2
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
import imageio


# moviefile = 'data/IMG_9391.MOV'
# ylims = (565,680)
# xlims = (400,585)
# rotate = True

# moviefile = 'data/IMG_9356.MOV'
# ylims = (365,452)
# xlims = (825,973)
# rotate = False

# moviefile = 'data/IMG_9411.MOV'
# ylims = (435,725)
# xlims = (760,1100)
# rotate = False

moviefile = 'data/IMG_9412.MOV'
ylims = (480,600)
xlims = (890,1040)
rotate = False


print("Importing video")
vidcap = cv2.VideoCapture(moviefile)
nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames = {nframes}")
print("Video imported")
nframes = 200

# Import the first frame to get dimensions
success,image = vidcap.read()

# Crop the image to the desired size
image = image[ylims[0]:ylims[1],xlims[0]:xlims[1]]

# Some of the videos get read in upside down for some reason.
if rotate: image = np.rot90(image,2)

# Plot the image and then update with the running average
plt.ion()
fg = plt.figure()
ax = fg.gca()
h = ax.imshow((image/np.max(image)*255).astype('uint8'))
plt.show()
plt.draw()

# open the file to write the animated gif
gif_file = "images/" + moviefile[5:-4] + "_mean_" + str(nframes) + ".gif"
with imageio.get_writer(gif_file, mode="I") as writer:
	mean = np.zeros_like(image)			# start with mean = all zeros
	for count in range(0,nframes):

		# average the latest frame with the older mean
		print(f"Averaging frame {count}")
		mean = (mean * count + image)/(count + 1)				
		
		# Create the next frame for the animated gif and write
		frame = (mean/np.max(mean)*255).astype('uint8')			
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		writer.append_data(frame)

		# Plot the new mean image on the screen
		h.set_data((mean/np.max(mean)*255).astype('uint8'))
		plt.draw()
		plt.pause(0.001)

		# import the next frame
		success,image = vidcap.read() 	
		if (not success): 				
			break

		# rotate and crop and go back to the start of the loop
		image = image[ylims[0]:ylims[1],xlims[0]:xlims[1]]
		if rotate: image = np.rot90(image,2)

# write to an image file
cv2.imwrite("images/" + moviefile[5:-4] + "_mean_" + str(nframes) + ".png", (mean/np.max(mean)*255).astype('uint8'))

# write final numpy array to a data file
np.save(moviefile[:-4] + "_mean_" + str(nframes),mean)
