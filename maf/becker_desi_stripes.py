import healpy as hp
import hpgeom as hpg
import numpy as np

Y3_MEDIAN_VISITS_SIM53_PER_BAND = {
    "u": 10,
    "g": 16,
    "r": 40,
    "i": 41,
    "z": 33,
    "y": 34,
}

try:
    from by_count_utils import FootprintCount, FootprintCountBasisFunction
    from rubin_scheduler.utils import DEFAULT_NSIDE, hpid2_ra_dec
    from survey_start import SURVEY_START_MJD

    def make_desi_footprints_for_y1_per_band(target_band, nside=DEFAULT_NSIDE, mjd_start=SURVEY_START_MJD, target_nvisits=None):
        """make a footprint basis function for the DESI observations for DESI Y1 only per band
        """

        target_nvisits = target_nvisits or Y3_MEDIAN_VISITS_SIM53_PER_BAND

        ra, dec = hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

        stripes = get_desi_stripes()
        result = np.zeros(ra.size, dtype=float)

        desired_counts = np.zeros(ra.size, dtype=float)

        for ramin, ramax, decmin, decmax, nu, year, name in zip(stripes["ramins"], stripes["ramaxs"],
                                            stripes["decmins"], stripes["decmaxs"],
                                            stripes["nus"], stripes["yrs"], stripes["names"]):

            if int(year) != 1 or name not in ["SGC-6", "SGC-7"]:
                continue

            indx = np.where((dec > decmin) & (dec < decmax) & (is_in_ra_range_degrees(ra, ramin, ramax)))[0]
            result[indx] = target_nvisits[target_band]
            desired_counts[indx] = target_nvisits[target_band]
        fpc = FootprintCount(mjd_start, nside=nside)
        fpc.set_footprint(target_band, result)
        bf = FootprintCountBasisFunction(bandname=target_band, nside=nside, footprint=fpc, mult_factor=1.,
                                        mask_reward_below=-4, mask_above_count=None,)

        return bf, desired_counts


    def make_desi_footprints(nside=DEFAULT_NSIDE, mjd_start=SURVEY_START_MJD, nu_overshoot_correction=None):
        """make a footprint basis function for the DESI observations.
        """

        nu_overshoot_correction = nu_overshoot_correction or {
            1: 0,
            2: 0,
            3: 5,
            4: 5,
            5: 5,
        }

        ra, dec = hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

        stripes = get_desi_stripes()
        result = np.zeros(ra.size, dtype=float)

        desired_counts = np.zeros(ra.size, dtype=float)

        for ramin, ramax, decmin, decmax, nu, year in zip(stripes["ramins"], stripes["ramaxs"],
                                            stripes["decmins"], stripes["decmaxs"],
                                            stripes["nus"], stripes["yrs"]):

            indx = np.where((dec > decmin) & (dec < decmax) & (is_in_ra_range_degrees(ra, ramin, ramax)))[0]
            # remove overshoot
            nu = nu - nu_overshoot_correction[int(year)]
            result[indx] = nu/year
            desired_counts[indx] = nu
        fpc = FootprintCount(mjd_start, nside=nside)
        fpc.set_footprint("u", result)
        bf = FootprintCountBasisFunction(bandname="u", nside=nside, footprint=fpc, mult_factor=1.,
                                        mask_reward_below=-4, mask_above_count=None,)

        return bf, desired_counts
except ImportError:
    print("Could not make DESI footprint functions!")


def is_in_ra_range_degrees(ra, low, high):
    _low = low % (360)
    _high = high % (360)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)


def get_desi_stripes(year_ordering: str = "swapped-1"):
    # code for DESI stripes from Anand Raichoor
    names = np.array([
        "NGC-1", "NGC-2", "NGC-3", "NGC-4", "NGC-5", "NGC-6", "NGC-7", "NGC-8", "NGC-9",
        "SGC-1", "SGC-2", "SGC-3", "SGC-4", "SGC-5", "SGC-6", "SGC-7", "SGC-8", "SGC-9",
    ])
    surveys = np.array([
        "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "Niji+VST", "Niji+VST", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS",
        "Niji+VST", "Niji+VST", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS", "POLU+IBIS",
    ])

    nus = [37.3, 32.4, 29.8, 29.1, 29.3, 24.6, 23.1, 20.4, 19.3,
           33.2, 28.4, 27.6, 25.8, 25.2, 22.8, 20.9, 20.8, 19.3]
    # original years
    if year_ordering == "original":
        yrs = np.array([
            4, 3, 2, 2, 4, 3, 1, 1, 5,
            4, 3, 3, 2, 2, 1, 1, 4, 5,
        ])
    elif year_ordering == "swapped-1":
        # swapped times for NGC "Niji+VST", "Niji+VST" fields w/ "POLU+IBIS" fields
        # reordered stripes for POU+IBIS to make bigger coherent areas
        # Y1 = 2029. Year stripe to be delivered
        yrs = np.array([
            4, 4, 3, 3, 2, 2, 1, 1, 5,
            4, 3, 3, 2, 2, 1, 1, 4, 5,
        ])
    else:
        raise RuntimeError(f"DESI year ordering `{year_ordering}` is not recognized!")

    ramins = np.array([
        121.01, 122.33, 123.69, 125.11, 126.60, 128.16, 129.81, 131.56, 133.43,
        315.57, 313.82, 312.19, 310.66, 309.21, 307.84, 306.53, 305.29, 304.10,
    ])
    ramaxs = np.array([
        255.71, 254.40, 253.00, 251.52, 249.93, 228.30, 225.58, 222.41, 218.59,
        44.37, 47.15, 49.56, 51.67, 53.53, 55.18, 56.64, 57.93, 59.06,
    ])
    decmins = np.array([
        11.375, 8.875, 5.875, 3.125, 0.375, -2.375, -5.375, -8.125, -10.875,
        1.625, -1.375, -4.125, -6.875, -9.625, -12.375, -15.375, -18.125, -20.875,
    ])
    decmaxs = np.array([
        15.125, 12.375, 9.375, 6.625, 3.875, 1.125, -1.625, -4.625, -7.375,
        5.125, 2.375, -0.625, -3.375, -6.125, -8.875, -11.625, -14.625, -17.375,
    ])
    return {
        "names": names,
        "surveys": surveys,
        "yrs": yrs,
        "ramins": ramins,
        "ramaxs": ramaxs,
        "decmins": decmins,
        "decmaxs": decmaxs,
        "nus": nus,
    }


def is_in_ra_range_radians(ra, low, high):
    _low = low % (2.0 * np.pi)
    _high = high % (2.0 * np.pi)
    if _low <= _high:
        return (ra >= _low) & (ra <= _high)
    else:
        return (ra >= _low) | (ra <= _high)


def does_it_work_for_desi(
    maps: dict[str, np.ndarray],
    desi_data: dict[str, np.ndarray] | None = None,
    ignore_sgc_1: bool = True,
    use_per_stripe_u_threshold: bool = False,
    tol: float = 0,
) -> (dict[str, bool], dict[str, dict[str, float]]):
    """Return a yes or no if a given survey will meet DESI requirements.

    Parameters
    ----------
    maps
        Dictionary mapping keys formed by Rubin year and band (e.g., "1 u", "3 g", etc.)
        to HEALPix maps of the number of visits (in ring order).
    desi_data
        Dictionary describing the DESI stripe properties as returned by the function
        `get_desi_stripes`.
    ignore_sgc_1
        If True, ignore the SGC-1 stripe from DESI as it falls outside of the current
        Rubin footprint.
    use_per_stripe_u_threshold
        If True, use the per stripe u band requirement. Otherwise, use 30 visits.
    tol 
        The DESI requirement is lowered by this tolerance value.

    Returns
    -------
    what_works
        Dict mapping DESI stripe name + band (e.g., "SGC-1 u") to boolean. Each value 
        is True if the median of the distribution of # of visits meets 
        the DESI requirement within `tol` in terms of number of visits.
    values
        Dict mapping DESI stripe name + band (e.g., "SGC-1 u") to dict of 
        the median # of visits and requirement.
    """

    what_works = {}
    values = {}
    nside = hpg.npixel_to_nside(next(iter(maps.values())).shape[0])

    for desi_ind in range(desi_data["yrs"].shape[0]):
        desi_year = int(desi_data["yrs"][desi_ind])
        rubin_year = desi_year + 1

        if desi_data["names"][desi_ind] == "SGC-1" and ignore_sgc_1:
            continue

        for band in ["u", "g", "r", "i", "z"]:
            mname = f"{rubin_year:d} {band}"

            pixels = hpg.query_box(
                nside,
                desi_data["ramins"][desi_ind],
                desi_data["ramaxs"][desi_ind],
                desi_data["decmins"][desi_ind],
                desi_data["decmaxs"][desi_ind],
                inclusive=True,
                nest=False,
                lonlat=True,
                degrees=True
            )

            data = maps[mname][pixels]
            value = float(np.percentile(data, 50.0))

            if band == "u":
                thresh = desi_data["nus"][desi_ind] if use_per_stripe_u_threshold else 30
            else:
                thresh = Y3_MEDIAN_VISITS_SIM53_PER_BAND[band]

            # for debugging
            # print(desi_data["names"][desi_ind], band, value, thresh, flush=True)
            key = f"{desi_data["names"][desi_ind]} {band}"

            what_works[key] = value >= thresh - tol
            values[key] = {"median": value, "requirement": thresh}

    return what_works, values
