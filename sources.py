import astropy.units as u
import numpy as np
from astroquery.sdss import SDSS
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy.coordinates import SkyCoord
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats
from regions import CirclePixelRegion, PixCoord


class Source:
    def __init__(self, source, count, WCS):
        self.position = pixel_to_skycoord(source['x'], source['y'], wcs = WCS).transform_to('icrs')
        self.radius = ((source['xmax'] - source['xmin'])/2) * 1.5
        self.source_id = count
        self.is_reference = False
        self.ref_mag = None
        self.ref_mag_err = None
        self.inst_mags = []
        self.inst_mag_errs = []
        self.calibrated_mags = []
        self.flagged = False
        self.weights = []
        self.errors = []
        self.chi2 = []

    def query_source(self, filter):
        search = SDSS.query_crossid(self.position, fields = ['ra', 'dec', f'psfMag_{filter}', f"psfMagErr_{filter}"], radius = 15*u.arcsec, region = False)

        if search:
            self.ref_mag = search[f'psfMag_{filter}']
            self.ref_mag_err = search[f'psfMagErr_{filter}']
            if search['type'] == 'STAR':
                self.is_reference = True


    def boundary_check(self, night):
        source_xy = SkyCoord.to_pixel(self.position, wcs = night.wcs[0])
        if (night.headers[0]['NAXIS1'] - source_xy[0]) < 0 or source_xy[0] < 0 or (night.headers[0]['NAXIS2'] - source_xy[1]) < 0 or source_xy[1] < 0:
            self.flagged = True
    
    def aperture_photometry(self, frame, obs_night):
        coords = SkyCoord.to_pixel(self.position, wcs = obs_night.wcs[0])
        pcoords = PixCoord(coords[0], coords[1])        

        inner_radius = self.radius
        inner_annulus_rad = self.radius + 5
        outer_annulus_rad = self.radius + 10


        source_circle = CirclePixelRegion(pcoords, inner_radius)
        source_circle_mask = source_circle.to_mask()
        source_aperture = source_circle_mask.cutout(frame)
        source_sum_unsub = np.sum(source_aperture)

        background_annulus = CircularAnnulus(coords, inner_annulus_rad, outer_annulus_rad)
        background_sum = aperture_photometry(frame, background_annulus)['aperture_sum'][0]

        source_flux_total = np.sum(source_aperture) - (source_circle.area/background_annulus.area)*background_sum

        readout_sum_source = source_circle.area*(obs_night.readout_noise**2)
        readout_sum_annulus = background_annulus.area*(obs_night.readout_noise**2)

        delta_n = (readout_sum_source + source_sum_unsub + ((source_circle.area/background_annulus.area)**2)*(readout_sum_annulus + background_sum))**(1/2)

        if source_flux_total < 0:
            self.flagged = True
        
        else:
            instrumental_mag = -2.5*np.log10(source_flux_total)
            instrumental_mag_error = 2.5*np.log10(np.e)*abs(delta_n/source_flux_total)
            self.inst_mags.append(instrumental_mag)
            self.inst_mag_errs.append(instrumental_mag_error)



    def add_calibrated_mag(self, mag):
        self.calibrated_mags.append(mag)
    
    def clear_calibration(self):
        self.weights = []
        self.errors = []
        self.calibrated_mags = []

    def add_chi(self, chi):
        self.chi2 = chi
    
    def add_error(self, err):
        self.errors.append(err)
        self.weights.append(1/(err**2))
    
    def get_info(self):
        print(f"Ra_Dec: {self.position}, Radius: {self.radius}, Is_Reference: {self.is_reference}, Reference_Magnitude: {self.ref_mag}, Average_Intstrumental_Magnitude: {np.mean(self.inst_mags)}, Calibrated_Magnitude: {np.mean(self.calibrated_mags)}, Flagged: {self.flagged}, ID: {self.source_id}")

"""
    def __iter__(self): #for writing out csv
        #return iter([self.position, self.is_reference, self.ref_mag, self.flagged, self.source_id, self.calibrated_mags])
        return iter([self.source_id, self.position, self.is_reference, self.ref_mag, self.flagged, self.calibrated_mags])
"""