import cv2
import sys
import numpy as np
import os

moviefile = 'data/IMG_9391.MOV'
ylims = (565,680)
xlims = (400,585)
rotate = True

# moviefile = 'data/IMG_9356.MOV'
# ylims = (365,452)
# xlims = (825,973)
# rotate = False

# moviefile = 'data/IMG_9411.MOV'
# ylims = (435,725)
# xlims = (760,1100)
# rotate = False

# moviefile = 'data/IMG_9412.MOV'
# ylims = (480,600)
# xlims = (890,1040)
# rotate = False


print("Importing video")
vidcap = cv2.VideoCapture(moviefile)
nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video imported")
print(f"Number of frames = {nframes}")

# Import the first frame to get dimensions
success,image = vidcap.read()

# Write the full frame to a file
cv2.imwrite("images/" + moviefile[5:-4] + "_full.png", image)

# Crop the image to the desired size
image = image[ylims[0]:ylims[1],xlims[0]:xlims[1]]

# Some of the videos get read in upside down for some reason.
if rotate: image = np.rot90(image,2)

# Find the shape of the memmap object that will need to be created
shape = tuple([nframes] + list(image.shape))

# Write the cropped and dynamically stretched images
cv2.imwrite("images/" + moviefile[5:-4] + "_cropped.png", image)
cv2.imwrite("images/" + moviefile[5:-4] + "_stretched.png", image/np.max(image)*255)

# Check to see if we've already processed this video
datafile = moviefile[:-4] + '.dat'
if not os.path.isfile(datafile):
    print("Writing video to .dat file")

    # Create a memmap to store the data for each frame
    frames = np.memmap(datafile,
                       mode='w+', 
                       shape = shape)
    count = 0;

    # for each frame in the file:   
    while success:
        print(f"Reading frame {count}")
        frames[count,:] = image         # store frame in memmap
        success,image = vidcap.read()   # read the next frame
        if (success):                   # if there is a next frame, then
            count += 1                  # add one to the count
        else:                           # if not, then
            break                       # we're done reading in frames

        #rotate and crop and go back to the top of the loop
        image = image[ylims[0]:ylims[1],xlims[0]:xlims[1]]
        if rotate: image = np.rot90(image,2)

    print("flushing .dat file")
    frames.flush()                  # flush and delete this memmap object
    del frames                      # (necessary because we want to bring
    print(".dat file written")      # the .dat file back in as read-only)

print("calculating mean and normalizing")
frames = np.memmap(datafile,        # create new memmap pointing to the 
                   mode='r',        # file we just wrote in the loop above
                   shape = shape)   # (or wrote earlier)

frames = frames[30:-30]             # drop first and last second (often shaky)
mean = np.memmap.mean(frames,axis=0)    # get the mean
mean = mean/np.max(mean)*255            # stretch it to the full dynamic range

# write to an image file
cv2.imwrite("images/" + moviefile[5:-4] + "_mean.png", mean)

# Do the same for median
median = np.median(frames,axis=0)   # get the mean
median = median/np.max(median)*255
cv2.imwrite("images/" + moviefile[5:-4] + "_median.png", median)

# write final numpy array to a data file
np.save(datafile[:-4] + "_mean",mean)
