__all__ = ("get_scheduler",)

import copy
import hashlib
import os
import pathlib

import numpy as np
import rubin_scheduler.scheduler.basis_functions as bf
import rubin_scheduler.scheduler.detailers as detailers
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.scheduler.schedulers import BaseQueueManager, CoreScheduler
from rubin_scheduler.scheduler.surveys import ScriptedSurvey
from rubin_scheduler.scheduler.utils import (
    CurrentAreaMap,
    Footprint,
    ScheduledObservationArray,
    make_rolling_footprints,
)
from rubin_scheduler.site_models import Almanac

# import lsst.ts.fbs.utils.maintel.lsst_surveys as lsst_surveys
# import lsst.ts.fbs.utils.maintel.roman_surveys as roman_surveys
# import lsst.ts.fbs.utils.maintel.too_surveys as too_surveys
import lsst_surveys as lsst_surveys
import roman_surveys as roman_surveys
import too_surveys as too_surveys
from survey_start import SURVEY_START_MJD


def generate_qm() -> BaseQueueManager:
    """Generate a QueueManager object."""

    detailer_list = []
    detailer_list.append(detailers.RotspUpdateDetailer())
    bf_list = []
    # This should get zenith masked
    bf_list.append(bf.SlewtimeBasisFunction())
    # Any clouds that have rolled in
    bf_list.append(bf.MapCloudBasisFunction())
    qm = BaseQueueManager(detailers=detailer_list, basis_functions=bf_list)

    return qm


def get_scheduler() -> tuple[int, CoreScheduler]:
    """Construct the LSST survey scheduler.

    The parameters are not accessible when calling as 'config'.

    Returns
    -------
    nside : `int`
        Healpix map resolution.
    scheduler : `rubin_scheduler.scheduler.scheduler.CoreScheduler`
        Feature based scheduler.
    """
    nside = 32
    science_program = "BLOCK-419"
    band_to_filter = {
        "u": "u_24",
        "g": "g_6",
        "r": "r_57",
        "i": "i_39",
        "z": "z_20",
        "y": "y_10",
    }
    exptime = 30
    nexp = 1
    u_exptime = 38
    u_nexp = 1

    template_exptime = 30
    u_template_exptime = 38

    survey_start_mjd = SURVEY_START_MJD

    # Safety mask parameters - constraints on all survey pointings
    # Generally shadow_minutes value is set by the survey, but can
    # be set here as well (will be overwritten if too short for survey).
    safety_mask_params = {
        "nside": nside,
        "wind_speed_maximum": 40,
        "apply_time_limited_shadow": False,
        "time_to_sunrise": 3.0,
        "min_az_sunrise": 150,
        "max_az_sunrise": 250,
    }

    safety_mask_params_ddf = copy.deepcopy(safety_mask_params)
    safety_mask_params_ddf["shadow_minutes"] = 30

    # General parameters for standard pairs (-80/80 default)
    camera_rot_limits = (-60.0, 60.0)
    pair_time = 33
    # Adjust these as the expected timing updates.
    # -- sets the expected time and number of pointings in a 'blob'.
    blob_survey_params = {
        "slew_approx": 8,
        "band_change_approx": 140.0,
        "read_approx": 3.07,
        "flush_time": 30.0,
        "smoothing_kernel": None,
        "nside": nside,
        "seed": 42,
        "twilight_scale": True,
    }
    # Seeing (FWHM in ") max for template
    fwhm_template_max = 1.2

    # Parameters for rolling cadence footprint definition
    nslice = 2  # N slices for rolling
    rolling_scale = 0.9  # Strength of rolling
    rolling_uniform = True  # Should we use the uniform rolling flag

    # Parameters for long-gaps survey
    nights_off = 3  # For long gaps

    # Parameters for near-sun twilight microsurvey
    ei_night_pattern = 4  # see pattern_dict below
    ei_bands = "riz"  # Bands to use for earth interior observations.
    ei_repeat = 4  # Number of times to repeat earth interior observations
    ei_am = 2.5  # Earth interior airmass limit
    ei_elong_req = 45.0  # Solar elongation required for inner solar system
    ei_area_req = 0.0  # Sky area required before attempting inner solar system

    # Mapping for night_pattern for near-sun twilight / twi_blob surveys.
    pattern_dict = {
        1: [True],
        2: [True, False],
        3: [True, False, False],
        4: [True, False, False, False],
        5: [True, True, True, True, False, False, False, False],
        6: [True, True, True, False, False, False, False],
        7: [True, True, False, False, False, False],
    }
    ei_night_pattern = pattern_dict[ei_night_pattern]

    # Generate footprint over the sky
    sky = CurrentAreaMap(nside=nside)
    footprints_hp_array, labels = sky.return_maps()
    # Identify pixels for rolling
    roll_indx = np.where((labels == "lowdust") | (labels == "virgo"))[0]
    roll_footprint = footprints_hp_array["r"] * 0
    roll_footprint[roll_indx] = 1

    footprints_hp = {}
    for key in footprints_hp_array.dtype.names:
        footprints_hp[key] = footprints_hp_array[key]

    # Set up a mask to contain some surveys within this region
    footprint_mask = footprints_hp["r"] * 0
    footprint_mask[np.where(footprints_hp["r"] > 0)] = 1

    # Use the Almanac to find the position of the sun at the start of survey
    almanac = Almanac(mjd_start=survey_start_mjd)
    sun_moon_info = almanac.get_sun_moon_positions(survey_start_mjd)
    sun_ra_start = sun_moon_info["sun_RA"].copy()

    # Define the rolling footprint
    footprints = make_rolling_footprints(
        fp_hp=footprints_hp,
        mjd_start=survey_start_mjd,
        sun_ra_start=sun_ra_start,
        nslice=nslice,
        scale=rolling_scale,
        nside=nside,
        wfd_indx=roll_indx,
        order_roll=1,
        n_cycles=3,
        uniform=rolling_uniform,
    )

    # Define the long-gaps (triplets) survey.
    gaps_night_pattern = [True] + [False] * nights_off
    long_gaps = lsst_surveys.gen_long_gaps_survey(
        footprints=footprints,
        nside=nside,
        camera_rot_limits=camera_rot_limits,
        exptime=exptime,
        nexp=nexp,
        u_exptime=u_exptime,
        u_nexp=u_nexp,
        pair_time=pair_time,
        night_pattern=gaps_night_pattern,
        science_program=science_program,
        blob_survey_params=blob_survey_params,
        safety_mask_params=safety_mask_params,
    )

    # This hash is provided by the script that
    # generates the pre-computed data. Execute it and paste
    # the provided value here.
    expected_hex_digest = "58950b2"
    pre_comp_file = (
        pathlib.Path(get_data_dir())
        / "scheduler"
        / f"ts_ddf_array_{expected_hex_digest}.npz"
    )
    if os.path.exists(pre_comp_file):
        loaded = np.load(pre_comp_file, allow_pickle=True)
        hash_object = hashlib.sha256()
        hash_object.update(loaded["hash_digest"])
        hex_digest = hash_object.hexdigest()[:7]
        if hex_digest == expected_hex_digest:
            obs_array_loaded = loaded["obs_array"]
            obs_array = ScheduledObservationArray(obs_array_loaded.size)
            for key in obs_array_loaded.dtype.names:
                obs_array[key] = obs_array_loaded[key]
        else:
            raise RuntimeError(
                f"Provided hash {expected_hex_digest} does not match loaded file hash {hex_digest}. "
                "Reach out for support so they can help you generate the correct file."
            )
        loaded.close()
    else:
        raise RuntimeError(
            f"Pre-computed DDF files {pre_comp_file} not available. "
            "Reach out for support so they can help execute the script "
            "that will generate this file before proceeding."
        )

    # Parameters for  DDF dithers
    camera_ddf_rot_limit = 55  # Rotator limit for DDF (degrees) .. 75
    camera_ddf_rot_per_visit = 3.0  # small rotation per visit (degrees) .. 3
    max_dither = 0.2  # Max radial dither for DDF (degrees)
    per_night = False  # Dither DDF per night (True) or per visit (False)

    band_expt = {
        "u": u_exptime,
        "g": exptime,
        "r": exptime,
        "i": exptime,
        "z": exptime,
        "y": exptime,
    }
    band_nexp = {"u": u_nexp, "g": nexp, "r": nexp, "i": nexp, "z": nexp, "y": nexp}
    u_nexp = band_nexp["u"]
    u_exptime = band_expt["u"]

    detailer_list = [
        detailers.CameraSmallRotPerObservationListDetailer(
            min_rot=-camera_ddf_rot_limit,
            max_rot=camera_ddf_rot_limit,
            per_visit_rot=camera_ddf_rot_per_visit,
        ),
        detailers.SplitDetailer(
            detailers.DitherDetailer(per_night=per_night, max_dither=max_dither),
            detailers.EuclidDitherDetailer(per_night=per_night),
        ),
        detailers.BandNexp(bandname="u", nexp=u_nexp, exptime=u_exptime),
        detailers.BandSortDetailer(),
        detailers.LabelRegionsAndDDFs(),
        detailers.TruncatePreTwiDetailer(),
    ]

    ddfs = [
        ScriptedSurvey(
            lsst_surveys.safety_masks(**safety_mask_params_ddf),
            nside=nside,
            detailers=detailer_list,
            survey_name="deep drilling",
            before_twi_check=False,
        )
    ]
    ddfs[0].set_script(obs_array)

    # Define the greedy surveys (single-visit per call)
    greedy = lsst_surveys.gen_greedy_surveys(
        nside=nside,
        camera_rot_limits=camera_rot_limits,
        exptime=exptime,
        nexp=nexp,
        u_exptime=u_exptime,
        u_nexp=u_nexp,
        footprints=footprints,
        science_program=science_program,
        safety_mask_params=safety_mask_params,
    )

    # Define the near-sun twilight microsurvey
    neo_micro = lsst_surveys.generate_twilight_near_sun(
        nside=nside,
        night_pattern=ei_night_pattern,
        max_airmass=ei_am,
        camera_rot_limits=camera_rot_limits,
        footprint_mask=footprint_mask,
        min_area=ei_area_req,
        bands=ei_bands,
        n_repeat=ei_repeat,
        max_elong=ei_elong_req,
        science_program=science_program,
        safety_mask_params=safety_mask_params,
    )

    # Define the standard pairs during the night survey
    blobs = lsst_surveys.generate_blobs(
        footprints=footprints,
        nside=nside,
        camera_rot_limits=camera_rot_limits,
        exptime=exptime,
        nexp=nexp,
        u_exptime=u_exptime,
        u_nexp=u_nexp,
        pair_time=pair_time,
        survey_start=survey_start_mjd,
        science_program=science_program,
        blob_survey_params=blob_survey_params,
        safety_mask_params=safety_mask_params,
    )

    # Define Roman scripted surveys
    roman_micro = [
        roman_surveys.gen_roman_on_season(
            nside=nside,
            max_dither=max_dither,
            per_night=per_night,
            camera_ddf_rot_limit=camera_ddf_rot_limit,
            camera_ddf_rot_per_visit=camera_ddf_rot_per_visit,
            exptimes=exptime,
            nexps=nexp,
            science_program=science_program,
            safety_mask_params=safety_mask_params,
        ),
        roman_surveys.gen_roman_off_season(
            nside=nside,
            max_dither=max_dither,
            per_night=per_night,
            camera_ddf_rot_limit=camera_ddf_rot_limit,
            camera_ddf_rot_per_visit=camera_ddf_rot_per_visit,
            exptimes=exptime,
            nexps=nexp,
            science_program=science_program,
            safety_mask_params=safety_mask_params,
        ),
    ]

    # Create template footprint.
    # Similar to rolling footprint but tracks visits separately
    # (only good seeing visits) and no rolling.
    template_fp = Footprint(survey_start_mjd, sun_ra_start, nside=nside)
    for key in footprints_hp_array.dtype.names:
        tmp_fp = np.where(footprints_hp_array[key] > 0, 1, np.nan)
        template_fp.set_footprint(key, tmp_fp)
    # Define template surveys
    template_surveys = lsst_surveys.gen_template_surveys(
        template_fp,
        nside=nside,
        band1s=["u", "g", "g", "r", "r", "i", "r", "z", "y"],
        band2s=["u", "g", "r", "r", "i", "z", "z", "y", "y"],
        seeing_fwhm_max=fwhm_template_max,
        camera_rot_limits=camera_rot_limits,
        exptime=template_exptime,
        nexp=nexp,
        u_exptime=u_template_exptime,
        u_nexp=u_nexp,
        n_obs_template={"u": 6, "g": 6, "r": 6, "i": 6, "z": 6, "y": 6},
        science_program=science_program,
        blob_survey_params=blob_survey_params,
        safety_mask_params=safety_mask_params,
    )

    # Define ToO surveys
    too_detailers = []
    too_detailers.append(
        detailers.CameraRotDetailer(
            min_rot=np.min(camera_rot_limits), max_rot=np.max(camera_rot_limits)
        )
    )
    too_detailers.append(detailers.BandSortDetailer())
    too_detailers.append(detailers.LabelRegionsAndDDFs())
    # Let's make a footprint to follow up ToO events
    too_footprint = np.where(footprints_hp["r"] > 0, 1.0, np.nan)

    toos = too_surveys.gen_too_surveys(
        nside=nside,
        detailer_list=too_detailers,
        too_footprint=too_footprint,
        n_snaps=nexp,
        science_program=science_program,
        safety_mask_params=safety_mask_params,
    )

    # Arrange the surveys in tiers.
    surveys = [
        toos,
        roman_micro,
        ddfs,
        template_surveys,
        long_gaps,
        blobs,
        neo_micro,
        greedy,
    ]

    qm = generate_qm()

    # Combine into CoreScheduler
    scheduler = CoreScheduler(
        surveys,
        nside=nside,
        survey_start_mjd=survey_start_mjd,
        band_to_filter=band_to_filter,
        queue_manager=qm,
    )

    return nside, scheduler


if __name__ == "config":
    nside, scheduler = get_scheduler()
