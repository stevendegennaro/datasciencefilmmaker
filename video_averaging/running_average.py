import cv2
import sys
import numpy as np
import os
from matplotlib import pyplot as plt
import imageio


def get_next_image(vidcap, BGR = True):
    # Import the first frame to get dimensions
    success,image = vidcap.read()
    if not success:
        return success,image
    if BGR:
        return success, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return succes, image


def import_movie(moviefile):
    print("Importing video")
    vidcap = cv2.VideoCapture(moviefile)
    nframes = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames = {nframes}")
    print("Video imported")
    nframes = 200
    return vidcap, nframes

def rotate_and_crop(image,xlims,ylims,rotate):
    # Some of the videos get read in upside down for some reason.
    if rotate: image = np.rot90(image,2)
    if xlims:
        image = image[:,xlims[0]:xlims[1]]
    if ylims:
        image = image[ylims[0]:ylims[1],:]
    return image

def average_frames(moviefile = 'data/Street Stream Recording.mp4',
                    xlims = None,
                    ylims = None,
                    rotate = False):

    vidcap, nframes = import_movie(moviefile)
    success, image = get_next_image(vidcap)
    image = rotate_and_crop(image,xlims,ylims,rotate)


    # Plot the image and then update with the running average
    plt.ion()
    fg = plt.figure()
    ax = fg.gca()
    h = ax.imshow((image/np.max(image)*255).astype('uint8'))
    plt.show()
    plt.draw()
    # sys.exit()

    # open the file to write the animated gif
    gif_file = "images/" + moviefile[5:-4] + "_mean_" + str(nframes) + ".gif"
    with imageio.get_writer(gif_file, mode="I") as writer:
        mean = np.zeros_like(image)         # start with mean = all zeros
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
            success, image = get_next_image(vidcap)
            if (not success):               
                break

            # rotate and crop and go back to the start of the loop
            image = rotate_and_crop(image,xlims,ylims,rotate)

    # write to an image file
    cv2.imwrite("images/" + moviefile[5:-4] + "_mean_" + str(nframes) + ".png", (mean/np.max(mean)*255).astype('uint8'))

    # write final numpy array to a data file
    np.save(moviefile[:-4] + "_mean_" + str(nframes),mean)
