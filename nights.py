"""
This class intitalizes the data for each observation night. It attempts to align all images and performs refernce source extraction and median template creation. 
Other house keeping operations are performed such as keeping track of observation filters used as well as date time information. 

HISTORY
    Created on 2023-05-22
        Luca Angeleri <lucaaldoa@gmail.com>
    Reworked on 2024-02-07
        Luca Angeleri <lucaaldoa@gmail.com>

"""


import astroalign as aa
import cv2
import glob
import numpy as np
import sep
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from joblib import Parallel, delayed


class Night:
    def __init__(self, file_path, night_number):
        self.path = file_path
        self.obs_night = night_number
        self.image_data = None
        self.headers = None
        self.wcs = None
        self.readout_noise = None
        self.aligned = None
        self.template = None
        self.references = None
        self.obs_filter = None
        self.no_obs_iamges = None
        self.mjd_times = []
        self.date_times = []
        self.date = []
        self.start_times = []
        self.is_aligned = True


    def align(self):
        reference_pixels = np.array([[100, 200], [1500, 1500], [1800, 3000]]).astype(np.float32) #these are arbitrary points spanning the image, should be done dynamically at some point
        reference_skycoords = []
        for point in reference_pixels:
            skyval = pixel_to_skycoord(point[1], point[0], wcs = self.wcs[0])
            reference_skycoords.append(skyval)
        
        target_points = []
        for i in range(len(self.image_data)):
            target_pixels = []
            for point in reference_skycoords:
                pixval = SkyCoord.to_pixel(point, wcs= self.wcs[i])
                target_pixels.append(pixval)
            target_points.append(np.array(target_pixels).astype(np.float32))


        affine_aligned = []
        for i, image in enumerate(self.image_data):
            M = cv2.getAffineTransform(reference_pixels, target_points[i])
            rows, cols = image.shape
            dst = cv2.warpAffine(image, M, (rows, cols))
            affine_aligned.append(dst.T)

        return affine_aligned
    



    def initialize_frames(self):
        science_dir = sorted(glob.glob(self.path + '/*'))
        #with ThreadPoolExecutor() as executor: #uses threading to open files in parallel
            #futures_results = executor.map(fits.open, science_dir)
        
        hdus = [fits.open(f) for f in science_dir]
        self.headers = [image[1].header for image in hdus]
        self.image_data = [image[1].data for image in hdus]
        self.wcs = [WCS(h) for h in self.headers]
        self.readout_noise = self.headers[0]["RDNOISE"]

        for header in self.headers:
            self.mjd_times.append(header['MJD-OBS'])
            self.date_times.append(header['DATE-OBS'])
            self.date.append(header['DAY-OBS'])
            self.start_times.append(['UTSTART'])
        
        try:
            self.aligned_images = self.align()
        except:
            print("Unable to align using affine transform. WCS matrix may be unavailble or points out of image bounds")
            try:
                self.aligned_images = [aa.register(image, self.image_data[0])[0] for image in self.image_data[0:]]
            except:
                print("Could not align images")
                self.aligned_images = self.image_data
                self.is_aligned = False
        
        self.template = np.median(self.aligned_images, axis = 0)
        background = sep.Background(self.template)
        self.references = sep.extract(self.template-background.back(), background.globalrms*3, minarea = 25, segmentation_map = False)
        self.obs_filter = self.headers[0]['filter'][0]
        self.no_obs_iamges = len(self.aligned_images)

    def get_info(self):
        print(f"Path: {self.path}, Observation Night: {self.obs_night}, Observation Date: {self.date[0]}, Aligned: {self.is_aligned}, Number of Reference Stars: {len(self.references)}, Filter: {self.obs_filter}")



