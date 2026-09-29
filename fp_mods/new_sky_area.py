import numpy as np
import healpy as hp
from rubin_scheduler.scheduler.utils import Phase3AreaMap


class SmallerFootprint(Phase3AreaMap):
    def __init__(self, dust_limit=0.08, **kwargs):
        super().__init__(dust_limit=dust_limit, **kwargs)

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
