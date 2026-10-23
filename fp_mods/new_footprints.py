__all__ = (
    "FootprintMod1",
    "FootprintMod2",
    "FootprintMod3",
    "FootprintMod4",
    "FootprintMod5",
    "FootprintMod6",
    "FootprintMod7",
    "FootprintMod8",
    "FootprintMod9",
)
import numpy as np
import healpy as hp
from astropy.io import fits
from rubin_scheduler.utils import angular_separation
import os
import copy
import json
from rubin_scheduler.data import get_data_dir
from matplotlib.path import Path

from rubin_scheduler.scheduler.utils import Phase3AreaMap
from astropy.coordinates import SkyCoord
from astropy import units as u

from rubin_scheduler.utils import _build_tree, _hpid2_ra_dec, _xyz_from_ra_dec


class NewBase(Phase3AreaMap):
    """Turn down the dust limit, Make sure SCP gets blocked off early"""

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


class FootprintMod6(NewBase):
    """Based on Knute's footprint from here:
    https://github.com/knutago/footprint-taskforce/blob/main/footprint-taskforce_GPmod_1.ipynb
    """

    def __init__(
        self,
        dust_limit=0.12,
        scp_dec_max=-66,
        bridge_width=8.43,
        bulgy_smooth_fwhm=6.2,
        bulgy_smooth_cut=0.074,
        **kwargs
    ):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)
        self.bridge_width = bridge_width
        self.bulgy_smooth_fwhm = np.radians(bulgy_smooth_fwhm)
        self.bulgy_smooth_cut = bulgy_smooth_cut

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):

        in_bridge = np.zeros(self.ra.size)

        good = np.where(
            (self.eclip_lat < self.bridge_width / 2.0)
            & (self.eclip_lat > -self.bridge_width / 2.0)
        )[0]
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
            "disk": ((-67.0, 50.1), (-2.0, 2.0)),
            "carina_ext": ((-79.0, -67.0), (-2.5, 2.0)),
            "bulge_bar_north": ((-10.0, 10.0), (2.0, 6.0)),
            "bulge_bar_south": ((-10.0, 10.0), (-6.0, -2.0)),
            "serpens_south": ((26.5, 30.0), (2.0, 4.5)),
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

        in_box = hp.sphtfunc.smoothing(in_box, fwhm=self.bulgy_smooth_fwhm)

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
    """Try smoothing the combined map"""

    def __init__(self, gal_priority_cut=1.72, gal_map_smooth_fwhm=3.0, **kwargs):
        super().__init__(gal_priority_cut=gal_priority_cut, **kwargs)
        self.gal_map_smooth_fwhm = np.radians(gal_map_smooth_fwhm)

    def read_galactic_plane_map(self, map_path):
        data_path = get_data_dir()
        map_file = os.path.join(data_path, map_path)
        hdul = fits.open(map_file)
        combined_map = copy.copy(hdul[1].data["combined_map"])
        hdul.close()

        combined_map = hp.sphtfunc.smoothing(
            combined_map, fwhm=self.gal_map_smooth_fwhm
        )

        return combined_map


class FootprintMod7(NewBase):
    """Try smoothing the combined map"""

    def __init__(self, dust_limit=0.12, **kwargs):
        super().__init__(dust_limit=dust_limit, **kwargs)

    def add_bulgy(self, band_ratios, label="bulgy"):
        in_bridge = np.zeros(self.ra.size)

        with open("street_fp.json", "r") as file:
            data = json.load(file)
            bridge_array = np.array(data["new_gp_pixels"])

        guess_nside = 64
        in_bridge = np.zeros(hp.nside2npix(guess_nside))
        in_bridge[bridge_array] = 1

        in_bridge = hp.ud_grade(in_bridge, self.nside)

        # Add dec less than dec max
        indx = np.where((in_bridge > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


class FootprintMod8(NewBase):
    """Try smoothing the combined map"""

    def __init__(
        self, dust_limit=0.12, filename="footprint_GPmod_nside64.npz", **kwargs
    ):
        super().__init__(dust_limit=dust_limit, **kwargs)

        load = np.load(filename)
        # self.footprints = copy.copy(load["footprints"])

        u_labels = np.unique(load["labels"])

        float_labels = np.zeros(load["labels"].size)
        for i, label in enumerate(u_labels):
            indx = np.where(load["labels"] == label)[0]
            float_labels[indx] = i

        float_labels = hp.ud_grade(float_labels, self.nside)

        self.labels = np.zeros(hp.nside2npix(self.nside), dtype="<U20")

        for i, label in enumerate(u_labels):
            indx = np.where(float_labels == i)
            self.labels[indx] = label

        load.close()

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):
        indx = np.where(self.labels == "bridge")[0]

        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]

    def add_bulgy(self, band_ratios, label="bulgy"):

        indx = np.where(
            (self.labels == label) | (self.labels == "bridge") | (self.labels == "rgps")
        )

        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]


class FootprintMod9(NewBase):
    """Make a shorter version of Knut's FootprintMod8"""

    def __init__(
        self,
        dust_limit=0.12,
        scp_dec_max=-66,
        bridge_width=14.0,
        bulgy_dec_max=0,
        stellar_density_max=2600,
        stellar_band="r",
        stellar_mag_limit=17.0,
        smooth_starfree_fwhm=2.0,
        gal_map_smooth_fwhm=8.0,
        gal_priority_cut=1.75,
        rgps_dec_max=0.,
        **kwargs
    ):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)
        self.bridge_width = bridge_width
        self.bulgy_dec_max = bulgy_dec_max

        self.gal_map_smooth_fwhm = np.radians(gal_map_smooth_fwhm)
        self.gal_priority_cut = gal_priority_cut
        self.rgps_dec_max = rgps_dec_max

        self.stellar_density_max = stellar_density_max
        self.stellar_band = stellar_band
        self.stellar_mag_limit = stellar_mag_limit

        if dust_limit > 0:
            self.find_dustfree()

        self.smooth_starfree_fwhm = smooth_starfree_fwhm
        self.starfree = np.ones(hp.nside2npix(self.nside))
        if stellar_density_max > 0:
            self.find_starfree()

    def find_starfree(self):
        """Load stellar density map and maks regions above density limit"""
        map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
        # Load the nside=64.
        # Note this in Galactic coords and nest order
        star_map_nside = 64
        filename = "TRIstarDensity_%s_nside_%i_ext.npz" % (
            self.stellar_band,
            star_map_nside,
        )
        load = np.load(os.path.join(map_dir, filename))
        star_map = load["starDensity"].copy()
        star_map_bins = load["bins"].copy()

        diff = np.abs(star_map_bins - self.stellar_mag_limit)
        bin_indx = np.where(diff == np.min(diff))[0]

        gal_l, gal_b = _hpid2_ra_dec(
            star_map_nside, np.arange(hp.nside2npix(star_map_nside)), nest=True
        )

        # Convert that to RA,dec. Then do nearest neighbor lookup.
        c = SkyCoord(l=gal_l * u.rad, b=gal_b * u.rad, frame="galactic").transform_to(
            "icrs"
        )
        ra = c.ra.rad
        dec = c.dec.rad

        self.tree = _build_tree(ra, dec)

        x, y, z = _xyz_from_ra_dec(np.radians(self.ra), np.radians(self.dec))
        dist, indices = self.tree.query(list(zip(x, y, z)))

        stellar_density_map = np.ravel(star_map[indices, bin_indx])

        self.starfree[np.where(stellar_density_map > self.stellar_density_max)[0]] = 0

        if self.smooth_starfree_fwhm > 0:
            self.starfree = hp.smoothing(self.starfree, fwhm=self.smooth_starfree_fwhm)
            self.starfree[np.where(self.starfree < 0.5)[0]] = 0
            self.starfree[np.where(self.starfree > 0)[0]] = 1

    def find_dustfree(self):
        dustfree = np.where(
            (self.dec > self.low_dust_dec_min)
            & (self.dec < self.low_dust_dec_max)
            & (self.low_dust == 1),
            1,
            0,
        )

        dustfree[np.where(self.low_dust == 0)] = 0

        if self.adjust_halves > 0:
            dustfree = np.where(
                (self.gal_lat < 0)
                & (self.dec > self.low_dust_dec_max - self.adjust_halves),
                0,
                dustfree,
            )

        self.dustfree = dustfree

    def add_lowdust_wfd(self, band_ratios, label="lowdust"):
        """Define a low-dust WFD region.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """

        indx = np.where((self.dustfree > 0) & (self.pix_labels == "") & (self.starfree > 0))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]

    def add_ecliptic_bridge(self, band_ratios, label="ecliptic_bridge"):

        in_bridge = np.zeros(self.ra.size)

        good = np.where(
            (self.eclip_lat < self.bridge_width / 2.0)
            & (self.eclip_lat > -self.bridge_width / 2.0)
        )[0]
        in_bridge[good] = 1

        # Add dec less than dec max
        indx = np.where((in_bridge > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]

    def add_dusty_plane(self, band_ratios, label="dusty_plane"):
        """Define high-dust region of the map.
        Updates self.healmaps and self.pix_labels.

        Parameters
        ----------
        band_ratios : `dict` {`str`: `float`}
            Dictionary of weights per band for the footprint.
        label : `str`, optional
            Label to apply to the resulting footprint
        """
        dusty = np.where(
            (
                (self.dec > self.dusty_dec_min)
                & (self.dec < self.dusty_dec_max)
                & ((self.low_dust == 0) | (self.starfree == 0))
            ),
            1,
            0,
        )

        indx = np.where((dusty > 0) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]

    def read_galactic_plane_map(self, map_path):
        data_path = get_data_dir()
        map_file = os.path.join(data_path, map_path)
        hdul = fits.open(map_file)
        combined_map = copy.copy(hdul[1].data["combined_map"])
        hdul.close()

        return combined_map


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

        return in_roman_footprint


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

        # Construcft a Roman footprint map
        roman_footprint = self.roman_footprint()

        indx_below = np.where(combined_map < self.gal_priority_cut)[0]
        combined_map[indx_below] = 0

        combined_map = hp.ud_grade(combined_map, self.nside)

        indx_cut = np.where(self.dec > self.rgps_dec_max)[0]
        combined_map[indx_cut] = 0
        roman_footprint[indx_cut] = 0

        indx_cut = np.where((self.dec > self.eclip_dec_min) & (self.ra < 180))[0]
        combined_map[indx_cut] = 0
        roman_footprint[indx_cut] = 0


        indx = np.where( ((combined_map > 0) | (roman_footprint > 0)) & (self.pix_labels == ""))
        self.pix_labels[indx] = label
        for bandname in band_ratios:
            self.healmaps[bandname][indx] = band_ratios[bandname]
