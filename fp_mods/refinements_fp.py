__all__ = ("Refine1", "Refine2")

from new_footprints import FootprintMod9
import numpy as np
import healpy as hp
from rubin_scheduler.utils import angular_separation



class Refine1(FootprintMod9):
    """Make some minor refinements using FootprintMod9 as a starting point
    """

    def __init__(self, roman_width=3., **kwargs):
        super().__init__(**kwargs)
        self.roman_width = roman_width

    def roman_footprint(self):

        in_roman_footprint = np.zeros(hp.nside2npix(self.nside))

        points = [
            # [100.90, 9.55, 3],
            # [84.92, -5.71, 3],
            [266.3, -29, 17],
            [279, -13, 10],
            [256, -45, 11],
            [155, -56.5, 6.5],
            [172, -62, 5],
            [190, -65, 5],
            [210, -64, 5],
            [242, -58, 6.5],
            [225, -60, 6.5],
            [220, -62, 8],  # Bonus manual point to get things connected!
        ]
        for point in points:
            dist = angular_separation(self.ra, self.dec, point[0], point[1])
            indx = np.where((dist < point[2]/2.))
            in_roman_footprint[indx] = 1

        thin_plane_indx = np.where((np.abs(self.gal_lat) < self.roman_width) &
                                   (self.gal_lon < 180))[0]
        in_roman_footprint[thin_plane_indx] = 1

        return in_roman_footprint

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):

        in_bridge = np.zeros(self.ra.size)

        good = np.where(
            (self.eclip_lat < self.bridge_width / 2.0)
            & (self.eclip_lat > -self.bridge_width / 2.0)
        )[0]
        in_bridge[good] = 1

        # Fill in the silly little hole
        hole_indx = np.where((self.ra > 280) & (self.ra < 300) &
                             (self.dec > -50) & (self.dec < -25))[0]
        in_bridge[hole_indx] = 1

        # Add dec less than dec max
        indx = np.where((in_bridge > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]

    def add_bulgy(
        self,
        band_ratios,
        label="bulgy",
        map_path="maps/GalacticPlanePriorityMaps/priority_GalPlane_footprint_map_data_sum.fits",
    ):
        """

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """

        # Read gal map
        combined_map = self.read_galactic_plane_map(map_path=map_path)

        combined_map = hp.sphtfunc.smoothing(
            combined_map, fwhm=self.gal_map_smooth_fwhm
        )

        # Construct a Roman footprint map
        roman_footprint = self.roman_footprint()

        indx_below = np.where(combined_map < self.gal_priority_cut)[0]
        combined_map[indx_below] = 0

        combined_map = hp.ud_grade(combined_map, self.nside)

        indx_cut = np.where(self.dec > self.low_dust_dec_max)[0]
        combined_map[indx_cut] = 0
        roman_footprint[indx_cut] = 0

        indx_cut = np.where((self.dec > self.eclip_dec_min) & (self.ra < 180))[0]
        combined_map[indx_cut] = 0
        roman_footprint[indx_cut] = 0

        indx = np.where(((combined_map > 0) | (roman_footprint > 0)) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


class Refine2(Refine1):

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
        # Turing down the bulge r-band 
        bulge_ratios={"u": 0.17, "g": 0.93, "r": 0.93, "i": 0.98, "z": 0.93, "y": 0.21},
        virgo_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        euclid_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        ecliptic_bridge_ratios={
            "u": 0.35,
            "g": 0.4,
            "r": 1.0,
            "i": 1.0,
            "z": 0.9,
            "y": 0.9,
        },
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


class Refine3(Refine1):

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
        # Turing down the bulge r-band 
        bulge_ratios={"u": 0.17, "g": 0.93, "r": 0.88, "i": 0.98, "z": 0.93, "y": 0.21},
        virgo_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        euclid_ratios={"u": 0.35, "g": 0.4, "r": 1.0, "i": 1.0, "z": 0.9, "y": 0.9},
        ecliptic_bridge_ratios={
            "u": 0.35,
            "g": 0.4,
            "r": 1.0,
            "i": 1.0,
            "z": 0.9,
            "y": 0.9,
        },
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