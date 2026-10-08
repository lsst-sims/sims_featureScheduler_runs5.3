__all__ = ("FootprintMod1", "FootprintMod2", "FootprintMod3", "FootprintMod4", "FootprintMod5",
           "FootprintMod6")
import numpy as np
import healpy as hp
from astropy.io import fits
from rubin_scheduler.utils import angular_separation
import os
import copy
from rubin_scheduler.data import get_data_dir

from astropy import units as u
from astropy.coordinates import SkyCoord
from rubin_scheduler.utils import _build_tree, _hpid2_ra_dec, xyz_from_ra_dec

from rubin_scheduler.scheduler.utils import Phase3AreaMap


class TrilegalStellarDensity(object):

    def __init__(self, band="r", nside=64, ext=True):
        self.map_dir = os.path.join(get_data_dir(), "maps", "TriMaps")
        self.band = band
        self.keynames = [
            f"starLumFunc_{self.band}",
            f"starMapBins_{self.band}",
        ]
        self.nside = nside
        self.ext = ext
        self.star_map = None

    def _read_map(self):
        if self.ext:
            filename = "TRIstarDensity_%s_nside_%i_ext.npz" % (
                self.band,
                self.nside,
            )
        else:
            filename = "TRIstarDensity_%s_nside_%i.npz" % (self.band, self.nside)
        star_map = np.load(os.path.join(self.map_dir, filename))
        self.star_map = star_map["starDensity"].copy()
        self.star_map_bins = star_map["bins"].copy()
        self.starmap_nside = hp.npix2nside(np.size(self.star_map[:, 0]))
        # Note, the trilegal maps are in galactic coordinates
        # and use nested healpix.
        gal_l, gal_b = _hpid2_ra_dec(self.nside, np.arange(hp.nside2npix(self.nside)), nest=True)

        # Convert that to RA,dec. Then do nearest neighbor lookup.
        c = SkyCoord(l=gal_l * u.rad, b=gal_b * u.rad, frame="galactic").transform_to("icrs")
        ra = c.ra.rad
        dec = c.dec.rad

        self.tree = _build_tree(ra, dec)

    def __call__(self, ra, dec):
        """ra, dec in degrees I think
        """
        if self.star_map is None:
            self._read_map()
        result = {}
        x, y, z = xyz_from_ra_dec(ra, dec)
        dist, indices = self.tree.query(list(zip(x, y, z)))

        result["starLumFunc_%s" % self.band] = self.star_map[indices, :]
        result["starMapBins_%s" % self.band] = self.star_map_bins
        return result



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


class FootprintMod6(NewBase):
    """Based on Knute's footprint from here:
    https://github.com/knutago/footprint-taskforce/blob/main/footprint-taskforce_GPmod_1.ipynb
    """

    def __init__(self, dust_limit=0.12, scp_dec_max=-66, stellar_n_limit=0.0002, 
                 stellar_mag_limit=17., band="r", **kwargs):
        super().__init__(dust_limit=dust_limit, scp_dec_max=scp_dec_max, **kwargs)

        self.stellar_n_limit = stellar_n_limit  # stars / arcsec^2 
        self.stellar_mag_limit = stellar_mag_limit
        # Should probably be stellar density band
        self.band = band

        self.tsd_obj = TrilegalStellarDensity(band=band)
        self.tsd = self.tsd_obj(self.ra, self.dec)

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
        dustfree = np.where(
            (self.dec > self.low_dust_dec_min) & (self.dec < self.low_dust_dec_max) & (self.low_dust == 1),
            1,
            0,
        )

        dustfree[np.where(self.low_dust == 0)] = 0

        if self.adjust_halves > 0:
            dustfree = np.where(
                (self.gal_lat < 0) & (self.dec > self.low_dust_dec_max - self.adjust_halves),
                0,
                dustfree,
            )

        bin_diff = self.tsd['starMapBins_%s' % self.band] - self.stellar_mag_limit
        stellar_bin_indx = np.min(np.where(np.abs(bin_diff) < 1e-6)[0])
        if np.size(stellar_bin_indx) == 0:
            raise ValueError("stellar_mag_limit = %f not valid" % self.stellar_mag_limit)

        low_stars = np.ones(dustfree.size)

        low_stars[np.where(self.tsd['starLumFunc_%s' % self.band][:, stellar_bin_indx] < self.stellar_n_limit)] = 0

        import pdb ; pdb.set_trace()
        # Where we meet both dust and stellar density cuts.
        # And nothing else has been labeled
        indx = np.where((dustfree > 0) & (low_stars > 0) & (self.pix_labels == ""))[0]

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


