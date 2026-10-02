__all__ = ("FootprintMod1", "FootprintMod2", "FootprintMod3", "FootprintMod4")
import numpy as np
import healpy as hp
from astropy.io import fits
from rubin_scheduler.utils import angular_separation
import os
import copy
from rubin_scheduler.data import get_data_dir

from rubin_scheduler.scheduler.utils import Phase3AreaMap


class NewBase(Phase3AreaMap):
    """Turn down the dust limit, Make sure SCP gets blocked off early
    """
    def __init__(self, dust_limit=0.11, scp_dec_max=-66, **kwargs):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)

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
        self.add_euclid_overlap(euclid_ratios)
        self.add_scp(scp_ratios)
        self.add_dusty_plane(dusty_plane_ratios)

        return self.healmaps, self.pix_labels


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

        data_path = get_data_dir()
        map_file = os.path.join(data_path, map_path)
        hdul = fits.open(map_file)
        combined_map = copy.copy(hdul[1].data["combined_map"])
        hdul.close()

        indx_below = np.where(combined_map < self.gal_priority_cut)[0]
        combined_map[indx_below] = 0

        combined_map = hp.ud_grade(combined_map, self.nside)

        # XXX--magic numbers to replace with proper limits
        # that are already set elsewhere
        indx_cut = np.where(self.dec > 10)[0]
        combined_map[indx_cut] = 0

        indx_cut = np.where((self.dec > 0) & (self.ra < 180))[0]
        combined_map[indx_cut] = 0

        indx = np.where((combined_map > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


class FootprintMod4(FootprintMod3):

    def __init__(self, gal_priority_cut=2.0, **kwargs):
        super().__init__(gal_priority_cut=gal_priority_cut, **kwargs)
        self.gal_priority_cut = gal_priority_cut
