import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.colors import LogNorm
import cv2
import os

def make_pictures():
    
    letter_array = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    for l in letter_array:
        
        
        PAD = np.load("CTMC_PAD_400_Minus_" + l + ".npy")
        
        nrow = 1; ncol = 1;
        fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    
        resolution = 0.01
        e_array = np.arange(resolution, 0.75, resolution)
        pos = plt.imshow(PAD, cmap='jet', aspect='auto', extent=[e_array.min(), e_array.max(), 0 + pi/8, 2*pi + pi/8],norm=LogNorm(vmin=pow(10, -2), vmax=1))
        plt.xlabel(r'$Energy$ in a.u.', fontsize=20)
        plt.ylabel(r'$\phi (radian)$', fontsize=18, rotation=90)
        # plt.colorbar(pos, ax=axs[0],fraction=0.046, pad=0.04)
        plt.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
    
        plt.savefig("images/PAD_" + l + ".png", bbox_inches='tight')
        # plt.show()

        plt.clf()

def Make_Video():
    image_folder = 'images'
    video_name = 'images/video2.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)
    
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()    
    
if __name__ == "__main__":
    make_pictures()
    Make_Video()
    
  