import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from rubin_scheduler.utils import DEFAULT_NSIDE, _hpid2_ra_dec

from rubin_scheduler.scheduler.utils.sky_area import CurrentAreaMap
from rubin_scheduler.scheduler.utils.footprints import (
    Footprint,
    StepSlopes,
    Footprints,
)

YEAR = 365.25


def _is_in_ra_range(ra, low, high):
    _low = low % (2.0 * np.pi)
    _high = high % (2.0 * np.pi)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)


def slice_quad_galactic_cut(
    target_map, wfd_indx=None, ra_range=None, swap_phase=True, desi_y1_only=False,
    ngc=True, sgc=True
):
    """
    Helper function for generating rolling footprints

    Parameters
    ----------
    target_map : dict of HEALpix maps
        The final desired footprint as HEALpix maps. Keys are band names
    wfd_indx : array of ints
        The indices of target_map that should be used for rolling.
        If None, assumes the rolling area should be where target_map['r'] == 1.
    ra_range : tuple of floats, optional
        If not None, then the indices are restricted to the given RA range
        in radians.
    """
    nside = hp.npix2nside(target_map["r"].size)
    ra, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad)
    _, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg

    indx_north = np.intersect1d(np.where(gal_lat >= 0)[0], wfd_indx)
    indx_south = np.intersect1d(np.where(gal_lat < 0)[0], wfd_indx)

    splits_north = slice_wfd_area_desi_north(target_map, wfd_indx=indx_north, desi_y1_only=desi_y1_only)
    splits_south = slice_wfd_area_desi_south(target_map, wfd_indx=indx_south, desi_y1_only=desi_y1_only)

    slice_indx = [[], []]
    if swap_phase:
        indx0 = 1
        indx1 = 0
    else:
        indx0 = 0
        indx1 = 1

    if desi_y1_only:
        if sgc:
            slice_indx[indx0].extend(
                indx_south[splits_south[0] : splits_south[1]].tolist()
            )
        if ngc:
            slice_indx[indx0].extend(
                indx_north[splits_north[0] : splits_north[1]].tolist()
            )
    else:
        if sgc:
            slice_indx[indx0].extend(
                indx_south[splits_south[0] : splits_south[1]].tolist()
                + indx_south[splits_south[2] : splits_south[3]].tolist()
            )
            slice_indx[indx1].extend(
                indx_south[splits_south[1] : splits_south[2]].tolist()
                + indx_south[splits_south[3] : splits_south[4]].tolist()
            )
        if ngc:
            slice_indx[indx0].extend(
                indx_north[splits_north[0] : splits_north[1]].tolist()
                + indx_north[splits_north[2] : splits_north[3]].tolist()
            )
            slice_indx[indx1].extend(
                indx_north[splits_north[1] : splits_north[2]].tolist()
                + indx_north[splits_north[3] : splits_north[4]].tolist()
            )

    if ra_range is not None:
        ra_indx = np.where(_is_in_ra_range(ra, *ra_range))[0]
        for j in range(2):
            slice_indx[j] = np.intersect1d(ra_indx, slice_indx[j])

    return slice_indx


def slice_wfd_area_desi_south(target_map, wfd_indx=None, desi_y1_only=False):

    nside = hp.npix2nside(target_map["r"].size)
    _, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indices = np.where(target_map["r"] == 1)[0]
    else:
        wfd_indices = wfd_indx
    wfd[wfd_indices] = 1

    # the second stripe in the north needs to cover DESI SGC stripes 4, 5, 6, 7
    # dec min and max of second strip
    if desi_y1_only:
        dec_max = (-8.875 + 1.7) / 180.0 * np.pi
    else:
        # dec_max = (-3.375 + 1.7) / 180.0 * np.pi  # <- covers DESI Y2
        dec_max = (2.375 + 1.7) / 180.0 * np.pi  # < covers DESI Y3
    dec_min = (-15.375 - 1.7) / 180.0 * np.pi
    msk = (dec[wfd_indices] <= dec_max) & (dec[wfd_indices] >= dec_min)
    start_ind = int(np.min(np.where(msk)[0]))
    end_ind = int(np.max(np.where(msk)[0])) + 1
    split_wfd_indices = [0, start_ind, end_ind]

    # rest of the cuts move to make area equal
    nslice = wfd_indices.shape[0] // 4
    split_wfd_indices.append(split_wfd_indices[-1] + 2*nslice - split_wfd_indices[1])
    split_wfd_indices.append(wfd_indices.shape[0])

    afracs = []
    for i in range(4):
        afracs.append(np.sum(wfd[wfd_indx[split_wfd_indices[i]:split_wfd_indices[i+1]]]) / np.sum(wfd))
        print("south:", i, np.sum(wfd[wfd_indx[split_wfd_indices[i]:split_wfd_indices[i+1]]]) / np.sum(wfd))

    print("south total and halves:", np.sum(afracs), afracs[0] + afracs[2], afracs[1] + afracs[3])

    if desi_y1_only:
        return split_wfd_indices[1:3]
    else:
        return split_wfd_indices


def slice_wfd_area_desi_north(target_map, wfd_indx=None, desi_y1_only=False):

    nside = hp.npix2nside(target_map["r"].size)
    _, dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    wfd = target_map["r"] * 0
    if wfd_indx is None:
        wfd_indices = np.where(target_map["r"] == 1)[0]
    else:
        wfd_indices = wfd_indx
    wfd[wfd_indices] = 1

    # the second stripe in the north needs to cover DESI NGC stripes 7, 8
    # dec min and max of second strip
    if desi_y1_only:
        dec_max = (-1.625 + 1.7) / 180.0 * np.pi
    else:
        # dec_max = (3.875 + 1.7) / 180.0 * np.pi <- covers DESI Y2
        dec_max = (9.375 + 1.7) / 180.0 * np.pi  # <- covers DESI Y3
    dec_min = (-8.125 - 1.7) / 180.0 * np.pi
    msk = (dec[wfd_indices] <= dec_max) & (dec[wfd_indices] >= dec_min)
    start_ind = int(np.min(np.where(msk)[0]))
    end_ind = int(np.max(np.where(msk)[0])) + 1
    split_wfd_indices = [0, start_ind, end_ind]

    # rest of the cuts move to make area equal
    nslice = wfd_indices.shape[0] // 4
    split_wfd_indices.append(split_wfd_indices[-1] + 2*nslice - split_wfd_indices[1])
    split_wfd_indices.append(wfd_indices.shape[0])

    afracs = []
    for i in range(4):
        afracs.append(np.sum(wfd[wfd_indx[split_wfd_indices[i]:split_wfd_indices[i+1]]]) / np.sum(wfd))
        #print("north:", i, np.sum(wfd[wfd_indx[split_wfd_indices[i]:split_wfd_indices[i+1]]]) / np.sum(wfd))

    #print("north total and halves:", np.sum(afracs), afracs[0] + afracs[2], afracs[1] + afracs[3])

    if desi_y1_only:
        return split_wfd_indices[1:3]
    else:
        return split_wfd_indices


# default function from commit e6d604718bcf3253434c8ddfee0250d9d09b5c8e
# on rubin_scheduler
def make_rolling_footprints(
    *,
    mjd_start,
    fp_hp=None,
    nside=DEFAULT_NSIDE,
    wfd_indx=None,
):
    """
    Generate rolling footprints

    Parameters
    ----------
    fp_hp : dict-like
        A dict with band name keys and HEALpix map values.
        Default None will load CurrentAreaMap. Assumes
        WFD is where r-band is 1.
    mjd_start : `float`
        The starting date of the survey.
    wfd_indx : array of ints
        The indices of the HEALpix map that are to be included in the rolling.

    Returns
    -------
    Footprints object
    """

    from rubin_scheduler.site_models import Almanac

    # Use the Almanac to find the position of the sun at the start of survey
    almanac = Almanac(mjd_start=mjd_start)
    sun_moon_info = almanac.get_sun_moon_positions(mjd_start)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    if fp_hp is None:
        sky = CurrentAreaMap(nside=nside)
        footprints, labels = sky.return_maps()
        fp_hp = {}
        for key in footprints.dtype.names:
            fp_hp[key] = footprints[key]

    hp_footprints = fp_hp

    nslice = 2
    scale = 0.9
    D = 1.0 - scale
    U = nslice - D * (nslice - 1)

    dvals = {
        0: "0",
        1: "1",
        D: "D",
        U: "U",
    }
    inv_dvals = {v: k for k, v in dvals.items()}

    # "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1",

    str_all_slopes = [
        # original rolling with flip fixed
        # "1 1 U D 1 U D 1 U D 1 1 1 1 1 1 1",
        # "1 1 D U 1 D U 1 D U 1 1 1 1 1 1 1",
        # "1 1 U D 1 U D 1 U D 1 1 1 1 1 1 1",
        # "1 1 D U 1 D U 1 D U 1 1 1 1 1 1 1",
        # "1 1 1 U D 1 U D 1 U D 1 1 1 1 1 1",
        # "1 1 1 D U 1 D U 1 D U 1 1 1 1 1 1",
        # "1 1 1 U D 1 U D 1 U D 1 1 1 1 1 1",
        # "1 1 1 D U 1 D U 1 D U 1 1 1 1 1 1",

        # delay by 0.25 years
        # "1 1 1 U D 1 U D 1 U D 1 1 1 1 1 1",
        # "1 1 1 D U 1 D U 1 D U 1 1 1 1 1 1",
        # "1 1 U D 1 U D 1 U D 1 1 1 1 1 1 1",
        # "1 1 D U 1 D U 1 D U 1 1 1 1 1 1 1",
        # "1 1 1 U D 1 U D 1 U D 1 1 1 1 1 1",
        # "1 1 1 D U 1 D U 1 D U 1 1 1 1 1 1",
        # "1 1 1 U D 1 U D 1 U D 1 1 1 1 1 1",
        # "1 1 1 D U 1 D U 1 D U 1 1 1 1 1 1",

        # # delay by 0.25 years and double first cadence up
        # "1 1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1",
        # "1 1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1",
        # "1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1 1",
        # "1 1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1",
        # "1 1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1",
        # "1 1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1",
        # "1 1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1",

        # delay by 0.25 years and double first cadence up
        # roll 1 season earlier
        "1 1 U U 1 D D 1 U D 1 1 1 1 1 1 1 1 1",
        "1 1 D D 1 U U 1 D U 1 1 1 1 1 1 1 1 1",
        "1 U U 1 D D 1 U D 1 1 1 1 1 1 1 1 1 1",
        "1 D D 1 U U 1 D U 1 1 1 1 1 1 1 1 1 1",
        "1 1 U U 1 D D 1 U D 1 1 1 1 1 1 1 1 1",
        "1 1 D D 1 U U 1 D U 1 1 1 1 1 1 1 1 1",
        "1 1 U U 1 D D 1 U D 1 1 1 1 1 1 1 1 1",
        "1 1 D D 1 U U 1 D U 1 1 1 1 1 1 1 1 1",

        # delay by 0.25 years and double first cadence up
        # roll 1 season earlier
        # flip phase of last season
        # "1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1 1",
        # "1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1 1 1",
        # "1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1 1 1",
        # "1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1 1",
        # "1 1 U U 1 D D 1 D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U 1 U D 1 1 1 1 1 1 1 1 1",

        # delay by 0.25 years and double first cadence up
        # roll 1 season earlier
        # add extra rolling season by removing uniform survey at Y6
        # "1 1 U U 1 D D D U D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U U D U D 1 1 1 1 1 1 1 1 1",
        # "1 U U 1 D D D U D U 1 1 1 1 1 1 1 1 1 1",
        # "1 D D 1 U U U D U D 1 1 1 1 1 1 1 1 1 1",
        # "1 1 U U 1 D D D U D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U U D U D 1 1 1 1 1 1 1 1 1",
        # "1 1 U U 1 D D D U D U 1 1 1 1 1 1 1 1 1",
        # "1 1 D D 1 U U U D U D 1 1 1 1 1 1 1 1 1",

    ]
    slice_names = [
        "a", "b", "a", "b", "a", "b", "a", "b", "a", "b"
    ]
    slice_quads = [
        270, 270, 360, 360, 90, 90, 180, 180
    ]
    slice_split_ind = [
        0, 1, 0, 1, 0, 1, 0, 1
    ]

    all_slopes = []
    for sstr in str_all_slopes:
        all_slopes.append(
            [inv_dvals[s] for s in sstr.split(" ")]
        )

    for i, s in enumerate(all_slopes):
        sname = "slice " + slice_names[i] + " for quad %3d"  % slice_quads[i]
        #print(sname + ": " + " ".join([dvals[x] for x in s]))

    fp_non_wfd = Footprint(mjd_start, sun_ra_start=sun_ra_start, nside=nside)
    rolling_footprints = []
    for i in range(len(all_slopes)):
        step_func = StepSlopes(rise=np.array(all_slopes[i], dtype=float))
        rolling_footprints.append(
            Footprint(
                mjd_start,
                sun_ra_start=sun_ra_start,
                step_func=step_func,
                nside=nside,
            )
        )

    wfd = hp_footprints["r"] * 0
    if wfd_indx is None:
        wfd_indx = np.where(hp_footprints["r"] == 1)[0]

    wfd[wfd_indx] = 1
    non_wfd_indx = np.where(wfd == 0)[0]

    quad_splits = {}
    quad_splits[270] = slice_quad_galactic_cut(
        hp_footprints,
        wfd_indx=wfd_indx,
        ra_range=(sun_ra_start + 1.5 * np.pi, sun_ra_start + 2.0 * np.pi),
    )
    quad_splits[360] = slice_quad_galactic_cut(
        hp_footprints,
        wfd_indx=wfd_indx,
        ra_range=(sun_ra_start, sun_ra_start + np.pi / 2),
    )
    quad_splits[90] = slice_quad_galactic_cut(
        hp_footprints,
        wfd_indx=wfd_indx,
        ra_range=(sun_ra_start + np.pi / 2, sun_ra_start + np.pi),
    )
    quad_splits[180] = slice_quad_galactic_cut(
        hp_footprints,
        wfd_indx=wfd_indx,
        ra_range=(sun_ra_start + np.pi, sun_ra_start + 1.5 * np.pi),
    )

    for key in hp_footprints:
        temp = hp_footprints[key] + 0
        temp[wfd_indx] = 0
        fp_non_wfd.set_footprint(key, temp)

        for i in range(len(all_slopes)):
            # make a copy of the current band
            temp = hp_footprints[key] + 0
            # Set the non-rolling area to zero
            temp[non_wfd_indx] = 0

            indx = quad_splits[slice_quads[i]][slice_split_ind[i]]

            # invert the indices
            ze = temp * 0
            ze[indx] = 1
            temp = temp * ze
            rolling_footprints[i].set_footprint(key, temp)


    result = Footprints([fp_non_wfd] + rolling_footprints)

    return result
