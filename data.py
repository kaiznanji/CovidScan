# Import libraries
import zipfile
import numpy as np
import os
import shutil
import math
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

def unzip():
    # Unzip data
    path = "raw_ct_scans\\"
    for files in os.listdir("raw_ct_scans"):
        # Move folder contents into proper classifications
        if any(x in files for x in ["0","1"]):
            with zipfile.ZipFile(path+files, "r") as z_fp:
                z_fp.extractall("covid")
        else:
            with zipfile.ZipFile(path+files, "r") as z_fp:
                z_fp.extractall("non_covid")
    main_path =  "CovidScan\\"
    for rdir in [main_path+"covid", main_path+"non_covid"]:
        filelist = []
        for tree,fol,fils in os.walk(rdir):
            filelist.extend([os.path.join(tree,fil) for fil in fils if fil.endswith('.gz')])
        for fil in filelist:
            os.rename(fil,os.path.join(rdir,fil[fil.rfind('\\') + 1:]))
        [os.remove(rdir+"\\"+path) for path in os.listdir(rdir)]


def prepare(mdir):
    array = []
    # Read file and obtain volume
    for path in os.listdir(mdir):
        data = nib.load(mdir+"\\"+path).get_fdata()
        # Normalize volume
        MIN_VOLUME = -1000
        MAX_VOLUME = 400
        data[data < MIN_VOLUME] = MIN_VOLUME
        data[data > MAX_VOLUME] = MAX_VOLUME
        normalized = (data - MIN_VOLUME) / (MAX_VOLUME - MIN_VOLUME)
        normalized = normalized.astype("float32")

        # Rotate image
        img = np.rot90(normalized)

        # Reshape image
        initial_depth = img.shape[2]
        initial_width = img.shape[0]
        initial_height = img.shape[1]
        
        IMG_SIZE = 64
        SLICES = 32

        depth_factor = SLICES / initial_depth
        height_factor = IMG_SIZE / initial_height
        width_factor = IMG_SIZE / initial_width

        new_img = zoom(img, (width_factor, height_factor, depth_factor))
        array.append(new_img)

    return array






    


