import healpy as hp
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
from rubin_scheduler.utils import DEFAULT_NSIDE, _hpid2_ra_dec


def desi_footprint(nside=DEFAULT_NSIDE):

    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
    galb = coord.galactic.b.deg

    desi_sgc = np.where(
        (dec <= np.radians(4))
        & (dec >= np.radians(-20))
        & (
            ((galb < np.radians(-25)) & (ra > np.radians(270)))
            | ((galb < np.radians(-45)) & (ra < np.radians(90)))
        ),
        1,
        0,
    )

    desi_sgc_y2 = np.where(
        (dec >= np.radians(-15)) & (dec <= np.radians(4)), desi_sgc, 0
    )
    desi_ngc = np.where(
        (dec <= np.radians(15))
        & (dec >= np.radians(-9))
        & (
            ((galb > np.radians(20)) & (ra < np.radians(180)))
            | ((galb > np.radians(28)) & (ra > np.radians(180)) & (dec >= 0))
            | ((galb > np.radians(44)) & (ra > np.radians(180)) & (dec <= 0))
        ),
        1,
        0,
    )
    desi_ngc_y2 = np.where(
        (dec <= np.radians(5)) & (dec >= np.radians(-9)), desi_ngc, 0
    )

    desi_y4 = desi_sgc + desi_ngc
    desi_y2 = desi_sgc_y2 + desi_ngc_y2

    return desi_y4, desi_y2
