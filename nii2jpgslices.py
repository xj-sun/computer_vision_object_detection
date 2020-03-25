"""
Script for converting nii to 2D slices in .jpg format in python




Author: Xujuan Sun / Yucheng Tang
Data: July 4, 2018

"""


import os
import os.path as path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing.pool import Pool
import nibabel as nib
import scipy
import scipy.misc

parser = argparse.ArgumentParser(
    description="Generate 2D slices from nii(nifity)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--volume_dir", default='',
    help="The folder that contains all the nii volumes"
)
parser.add_argument(
    "--image_dir", default='',
    help="The folder that contains all 2D slices"
)
parser.add_argument(
    "--num_workers", type=int, default=11,
    help="Number of processing workers"
)
args = parser.parse_args()

nii_files = [
    d for d in os.listdir(args.volume_dir) if d.endswith('.nii')
]
count = 0
total_len = len(nii_files)
for nii in nii_files:
    count += 1
    print("[{}/{}] Converting {}".format(count, total_len, nii))
    output_dir = os.path.join(args.image_dir, nii)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    nii_path = os.path.join(args.volume_dir, nii)
    img = nib.load(nii_path)
    img_array = np.array(img.get_fdata())
    for i in range(img_array.shape[2]):
        cur_slice = img_array[:,:,i]
        cur_slice_file = os.path.join(output_dir, "slice_{}.png".format(str(i).zfill(4)))
        cur_slice_image = scipy.misc.toimage(cur_slice, high = np.max(cur_slice), 
            low=np.min(cur_slice),mode='L')        
        cur_slice_image.save(cur_slice_file)
print("Done")

