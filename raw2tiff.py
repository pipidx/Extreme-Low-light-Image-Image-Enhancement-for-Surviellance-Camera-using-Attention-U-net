from __future__ import division
from __future__ import print_function
import os, scipy.io
import glob
import numpy as np
import rawpy
import tifffile

input_dirs = ['./dataset/Sony/short', './dataset/Sony/long']
output_dirs = ['./dataset/short', './dataset/long']


for i, input_dir in enumerate(input_dirs):
    if not os.path.isdir(input_dir):
        print('input path not exits')
        exit(1)

    if not os.path.isdir(output_dirs[i]):
        os.makedirs(output_dirs[i])

    raw_filelist = glob.glob(os.path.join(input_dir, '*.ARW'))

    for raw_file in raw_filelist:  
        raw = rawpy.imread(raw_file)
        processed = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)

        filename = os.path.splitext(os.path.split(raw_file)[-1])[0]
        tiff_file = os.path.join(output_dirs[i], filename+'.tiff')
        tifffile.imwrite(tiff_file, data=processed)
        print('saved:', tiff_file)
