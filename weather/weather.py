import argparse
import os
import subprocess
import sys

import numpy as np
import numpy.typing as npt
import rubin_scheduler
from rubin_scheduler.scheduler import sim_runner
from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.scheduler.schedulers import CoreScheduler, SimpleBandSched
from rubin_scheduler.scheduler.targetofo import gen_all_events
from rubin_scheduler.scheduler.utils import ObservationArray
from rubin_scheduler.utils import DEFAULT_NSIDE, mjd2dayobs
from lsst_survey_sim.lsst_support import survey_times

from fbs_config import SURVEY_START_MJD, get_scheduler

EXPECTED_WAIT_SETTLE = 3.0
CURRENT_TMA_DEFAULT = {
    "azimuth_maxspeed": 2.0,
    "azimuth_accel": 2.0,
    "azimuth_jerk": 8.0,
    "altitude_maxspeed": 2.0,
    "altitude_accel": 2.0,
    "altitude_jerk": 8.0,
    "settle_time": EXPECTED_WAIT_SETTLE,
}


def set_run_info(
    dbroot: str | None = None, file_end: str = "", out_dir: str = ".",
    cloud_offset_year: int = 0
) -> tuple[str, dict]:
    """Gather versions of software used to record"""
    extra_info = {}
    exec_command = ""
    for arg in sys.argv:
        exec_command += " " + arg
    extra_info["exec command"] = exec_command
    try:
        extra_info["git hash"] = subprocess.check_output(["git", "rev-parse", "HEAD"])
    except subprocess.CalledProcessError:
        extra_info["git hash"] = "Not in git repo"

    extra_info["file executed"] = os.path.realpath(__file__)
    try:
        rs_path = rubin_scheduler.__path__[0]
        hash_file = os.path.join(rs_path, "../", ".git/refs/heads/main")
        extra_info["rubin_scheduler git hash"] = subprocess.check_output(
            ["cat", hash_file]
        )
    except subprocess.CalledProcessError:
        pass

    # Use the filename of the script to name the output database
    if dbroot is None:
        fileroot = os.path.basename(sys.argv[0]).replace(".py", "") + "_"
    else:
        fileroot = dbroot + "_"
    fileroot = os.path.join(out_dir, fileroot + "cloudso%i" % cloud_offset_year + file_end)
    return fileroot, extra_info


def make_observatory(
    nside=DEFAULT_NSIDE,
    survey_start_mjd=SURVEY_START_MJD,
    sim_to_o=None,
    readtime: float = 3.07,
    band_changetime: float = 120.0,
    cloud_offset_year: float = 0.,
):

    survey_info = survey_times(
        downtime_start_day_obs=int(mjd2dayobs(survey_start_mjd)),
        new_downtime_ndays=3700,
        random_seed=55,
        minutes_after_sunset12=0,
        early_dome_closure=0,
        add_downtime=True,
        real_downtime=False,
        visits=None,
        survey_start_mjd=survey_start_mjd,
    )

    observatory = ModelObservatory(
        nside=nside,
        mjd_start=survey_start_mjd,
        sim_to_o=sim_to_o,
        downtimes=survey_info["downtimes"],
        cloud_offset_year=cloud_offset_year,
    )

    tma_kwargs = CURRENT_TMA_DEFAULT
    observatory.setup_telescope(**tma_kwargs)
    observatory.setup_camera(band_changetime=band_changetime, readtime=readtime)
    # Remove close-loop optics iterations
    observatory.observatory.setup_optics(
        cl_delay=[0.0, 0.0], cl_altlimit=[0.0, 9.0, 90.0]
    )

    return observatory


def run_sched(
    scheduler: CoreScheduler,
    observatory: ModelObservatory,
    survey_length: float = 365.25,
    nside: int = DEFAULT_NSIDE,
    filename: str | None = None,
    verbose: bool = False,
    extra_info: dict | None = None,
    illum_limit: float = 40.0,
    event_table: npt.NDArray | None = None,
    snapshot_dir: str | None = None,
) -> tuple[ModelObservatory, CoreScheduler, ObservationArray]:
    """Run survey"""
    n_visit_limit = None
    fs = SimpleBandSched(illum_limit=illum_limit)

    observatory, scheduler, observations = sim_runner(
        observatory,
        scheduler,
        sim_duration=survey_length,
        filename=filename,
        delete_past=True,
        n_visit_limit=n_visit_limit,
        verbose=verbose,
        extra_info=extra_info,
        band_scheduler=fs,
        event_table=event_table,
        snapshot_dir=snapshot_dir,
    )

    return observatory, scheduler, observations


def sched_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Print more output"
    )
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--survey_length", type=float, default=365.25 * 10, help="Survey length in days"
    )
    parser.add_argument("--out_dir", type=str, default="", help="Output directory")
    parser.add_argument("--dbroot", type=str, help="Database root")
    parser.add_argument(
        "--setup_only",
        dest="setup_only",
        default=False,
        action="store_true",
        help="Only construct scheduler, do not simulate",
    )
    parser.add_argument(
        "--snapshot_dir",
        type=str,
        default="",
        help="Directory for scheduler snapshots.",
    )

    parser.add_argument("--cloud_offset_year", type=float, default=0.)

    return parser


if __name__ == "__main__":

    parser = sched_argparser()
    args = parser.parse_args()

    cloud_offset_year = args.cloud_offset_year

    dbroot = args.dbroot
    out_dir = args.out_dir
    fileroot, extra_info = set_run_info(
        dbroot=dbroot,
        file_end="v5.3.0_",
        out_dir=out_dir,
        cloud_offset_year = cloud_offset_year
    )
    years = np.round(args.survey_length / 365.25)
    filename = os.path.join(fileroot + "%iyrs.db" % years)

    nside, scheduler = get_scheduler()

    too_scale = 1.0
    sim_ToOs, event_table = gen_all_events(scale=too_scale, nside=nside)

    observatory = make_observatory(sim_to_o=sim_ToOs, cloud_offset_year=cloud_offset_year)

    observatory, scheduler, observations = run_sched(
        scheduler,
        observatory,
        survey_length=args.survey_length,
        nside=nside,
        filename=filename,
        verbose=args.verbose,
        event_table=event_table,
        extra_info=extra_info,
    )
