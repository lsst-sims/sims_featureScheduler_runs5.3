__all__ = ("FootprintMod1", "FootprintMod2", "FootprintMod3", "FootprintMod4", "FootprintMod5",
           "FootprintMod6")
import numpy as np
import healpy as hp
from astropy.io import fits
from rubin_scheduler.utils import angular_separation
import os
import copy
from rubin_scheduler.data import get_data_dir
from matplotlib.path import Path

from rubin_scheduler.scheduler.utils import Phase3AreaMap


class NewBase(Phase3AreaMap):
    """Turn down the dust limit, Make sure SCP gets blocked off early
    """
    def __init__(self, dust_limit=0.11, scp_dec_max=-66, **kwargs):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):
        pass

    def return_maps(
        self,
        magellenic_clouds_ratios={
            "u": 0.65,
            "g": 0.65,
            "r": 1.1,
            "i": 1.1,
            "z": 0.34,
            "y": 0.35,
        },
        scp_ratios={"u": 0.1, "g": 0.175, "r": 0.1, "i": 0.135, "z": 0.046, "y": 0.047},
        nes_ratios={"g": 0.255, "r": 0.33, "i": 0.33, "z": 0.23},
        dusty_plane_ratios={
            "u": 0.093,
            "g": 0.26,
            "r": 0.26,
            "i": 0.26,
            "z": 0.26,
            "y": 0.093,
        },
        low_dust_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        bulge_ratios={"u": 0.17, "g": 0.93, "r": 0.98, "i": 0.98, "z": 0.93, "y": 0.21},
        virgo_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        euclid_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        ecliptic_bridge_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
    ):
        # Array to hold the labels for each pixel
        self.pix_labels = np.zeros(hp.nside2npix(self.nside), dtype="U20")
        self.healmaps = np.zeros(
            hp.nside2npix(self.nside),
            dtype=list(zip(["u", "g", "r", "i", "z", "y"], [float] * 7)),
        )

        # Note, order here matters.
        # Once a HEALpix is set and labled, subsequent add_ methods
        # will not override that pixel.
        self.add_magellanic_clouds(magellenic_clouds_ratios)
        self.add_lowdust_wfd(low_dust_ratios)
        self.add_virgo_cluster(virgo_ratios)
        self.add_bulgy(bulge_ratios)
        self.add_nes(nes_ratios)
        self.add_ecliptic_bridge(ecliptic_bridge_ratios)
        self.add_euclid_overlap(euclid_ratios)
        self.add_scp(scp_ratios)
        self.add_dusty_plane(dusty_plane_ratios)

        return self.healmaps, self.pix_labels


class FootprintMod6(NewBase):
    """Based on Knute's footprint from here:
    https://github.com/knutago/footprint-taskforce/blob/main/footprint-taskforce_GPmod_1.ipynb
    """

    def __init__(self, dust_limit=0.12, scp_dec_max=-66, bridge_width=8.43, 
                 bulgy_smooth_radius=6.2, bulgy_smooth_cut=0.074, **kwargs):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)
        self.bridge_width = bridge_width
        self.bulgy_smooth_radius = np.radians(bulgy_smooth_radius)
        self.bulgy_smooth_cut = bulgy_smooth_cut

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):

        in_bridge = np.zeros(self.ra.size)

        good = np.where((self.eclip_lat < self.bridge_width/2.) &
                        (self.eclip_lat > -self.bridge_width/2.))[0]
        in_bridge[good] = 1

        # Add dec less than dec max
        indx = np.where((in_bridge > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


    def add_bulgy(self, band_ratios, label="bulgy"):
        """Define a bulge region,

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """

        # RGPS wide-field boxes (Galactic l, b in deg; l in [-180, 180]), from
        # roman-docs.stsci.edu/roman-community-defined-surveys/galactic-plane-survey (691.15 deg^2)
        rgps_boxes = {
            "disk":            ((-67.0, 50.1), (-2.0, 2.0)),
            "carina_ext":      ((-79.0, -67.0), (-2.5, 2.0)),
            "bulge_bar_north": ((-10.0, 10.0), (2.0, 6.0)),
            "bulge_bar_south": ((-10.0, 10.0), (-6.0, -2.0)),
            "serpens_south":   ((26.5, 30.0), (2.0, 4.5)),
        }

        # Track if points are in the box
        # This will probably fail if we try to do things
        # at the 180,-180 logitude border.
        in_box = np.zeros(self.ra.size)

        wrap_lon = self.gal_lon + 0
        wrap_lon[np.where(wrap_lon > 180)] -= 360

        points_to_check = np.array([wrap_lon, self.gal_lat]).T

        # Loop over each box
        for key in rgps_boxes:
            corners = [
                       [rgps_boxes[key][0][0], rgps_boxes[key][1][0]],
                       [rgps_boxes[key][0][1], rgps_boxes[key][1][0]],
                       [rgps_boxes[key][0][1], rgps_boxes[key][1][1]],
                       [rgps_boxes[key][0][0], rgps_boxes[key][1][1]],
                       ]
            box_path = Path(np.array(corners))
            indx_path = box_path.contains_points(points_to_check)
            in_box[indx_path] = 1

        in_box = hp.sphtfunc.smoothing(in_box, fwhm=self.bulgy_smooth_radius)
    
        # Do not go far north
        in_box[np.where(self.dec > self.dusty_dec_max)] = 0

        
        indx = np.where((in_box > self.bulgy_smooth_cut) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]



class FootprintMod1(NewBase):
    def __init__(self, dust_limit=0.11, **kwargs):
        super().__init__(dust_limit=dust_limit, **kwargs)


class FootprintMod2(NewBase):

    def add_bulgy(self, band_ratios, label="bulgy"):
        """Define a bulge region, where the 'bulge' is a series of
        circles set by points defined to match as best as possible the
        map requested by the SMWLV working group on galactic plane coverage.
        Implemented in v3.0.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        # Some RA, dec, radius points that
        # seem to cover the areas that are desired
        points = [
            #[100.90, 9.55, 3],
            #[84.92, -5.71, 3],
            [266.3, -29, 17],
            [279, -13, 10],
            [256, -45, 11],
            [155, -56.5, 6.5],
            [172, -62, 5],
            [190, -65, 5],
            [210, -64, 5],
            [242, -58, 6.5],
            [225, -60, 6.5],
        ]
        for point in points:
            dist = angular_separation(self.ra, self.dec, point[0], point[1])
            # Only change pixels where the label isn't already set.
            indx = np.where((dist < point[2]) & (self.pix_labels == ""))
            self.pix_labels[indx] = label
            for bandname in band_ratios:
                self.healmaps[bandname][indx] = band_ratios[bandname]


class FootprintMod3(NewBase):

    def __init__(self, gal_priority_cut=1.503, **kwargs):
        super().__init__(**kwargs)
        self.gal_priority_cut = gal_priority_cut

    def read_galactic_plane_map(self, map_path):
        data_path = get_data_dir()
        map_file = os.path.join(data_path, map_path)
        hdul = fits.open(map_file)
        combined_map = copy.copy(hdul[1].data["combined_map"])
        hdul.close()
        return combined_map

    def add_bulgy(self, band_ratios, label="bulgy",
                  map_path='maps/GalacticPlanePriorityMaps/priority_GalPlane_footprint_map_data_sum.fits'):
        """Define a bulge region, where the 'bulge' is a series of
        circles set by points defined to match as best as possible the
        map requested by the SMWLV working group on galactic plane coverage.
        Implemented in v3.0.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """

        combined_map = self.read_galactic_plane_map(map_path=map_path)

        indx_below = np.where(combined_map < self.gal_priority_cut)[0]
        combined_map[indx_below] = 0

        combined_map = hp.ud_grade(combined_map, self.nside)

        indx_cut = np.where(self.dec > self.dusty_dec_max)[0]
        combined_map[indx_cut] = 0

        indx_cut = np.where((self.dec > self.eclip_dec_min) & (self.ra < 180))[0]
        combined_map[indx_cut] = 0

        indx = np.where((combined_map > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


class FootprintMod4(FootprintMod3):

    def __init__(self, gal_priority_cut=2.0, **kwargs):
        super().__init__(gal_priority_cut=gal_priority_cut, **kwargs)
        self.gal_priority_cut = gal_priority_cut


class FootprintMod5(FootprintMod3):
    """Try smoothing the combined map
    """
    def __init__(self, gal_priority_cut=1.65, gal_map_smooth_fwhm=3., **kwargs):
        super().__init__(gal_priority_cut=gal_priority_cut, **kwargs)
        self.gal_map_smooth_fwhm = np.radians(gal_map_smooth_fwhm)

    def read_galactic_plane_map(self, map_path):
        data_path = get_data_dir()
        map_file = os.path.join(data_path, map_path)
        hdul = fits.open(map_file)
        combined_map = copy.copy(hdul[1].data["combined_map"])
        hdul.close()

        combined_map = hp.sphtfunc.smoothing(combined_map, fwhm=self.gal_map_smooth_fwhm)

        return combined_map


