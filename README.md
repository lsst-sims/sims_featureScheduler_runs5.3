# sims_featureScheduler_runs5.3
The ever expanding simulations


v5.3 vs v5.0

v5.0 - no template tier
v5.3 - template tier 5 per band (u & g are same-filter pairs, g-r-i-z-y are (can be) mixed filter pairs, plus y-y), 25 minute pairs, two hour angle limits (+/-2.5 hours for standard (50sq deg), +/-1.25 hours for cleanup small area (10 sq deg))

v5.0 - pairs at 33 minutes and 15 minutes (ugrizy in mixed filters; y-y as well) (15 minute pairs only rizy)
v5.3 - dynamically time-variable pairs ranging from 15(?) minutes to 40 minutes (ugrizy in mixed filters; y-y as well)

v5.0 - ocean sequences in DDFs, COSMOS and EDFS_ab deep in Y1
v5.3 - (accordion? bugfixed?) ocean sequences in DDFs, only COSMOS deep in Y1 -- double check sequences against v5.1 

v5.0 - ToO surveys
v5.3 - ToO surveys with improved detailers for coverage plus bugfix for more ToOs (check time in ToOs compared to v5.0)

v5.0 and v5.3 - no footprint changes

v5.0 - start date Nov 1, 2025
v5.3 - start date June 15, 2026 


v5.0 - 22 weeks (2 weeks per year) scheduled downtime, UnscheduledDowntimeMoreY1Data unscheduled downtime (70% to 99% over Y1)

v5.3 - new downtime
New planned downtime - 46 weeks of block scheduled downtime + 18 weeks of single night engineering time
Unscheduled downtime - 80% rising to ~97% (or so) after Y2 for short periods of downtime + old 21 weeks of unscheduled downtime

v5.0 - slew performance 40% TMA + 3s settle
v5.3 - slew performance 20% TMA + 3s settle (+scatter?), moving to 20% TMA + 1.5s settle after Y1 + reduced scatter


Requested sims: 
* new baseline configuration + old perf
* new baseline configuration + new perf
* rolling after year 2 + new perf
* desi + new perf
* template tier with shorter exposure 20s + new perf


==== 

New downtime: 
```
survey_info = lsst_survey_sim.survey_times(
    downtime_start_day_obs: int,
    new_downtime_ndays: float = 365.0,
    random_seed: int = 55,
    minutes_after_sunset12: float = 0,
    early_dome_closure: float = 0,
    add_downtime: bool = True,
    real_downtime: bool = False,
    visits: pd.DataFrame | None = None,
    survey_start_mjd: float = SURVEY_START_MJD,
) -> dict:

downtime_start_day_obs = dayobs of start of survey (use one of our functions for time to dayobs)
new_downtime_ndays = 3700
minutes_after_sunset12 = 0  (it's actually ~15 right now; alternatively use 10 for Y1 and 0 after)
early_dome_close = 0
add_downtime = True
real_downtime = False
visits = None
survey_start_mjd = survey_start_mjd 
```

survey_info['downtimes'] => into model observatory

(replaces downtimes from ScheduledDowntimeData and UnscheduledDowntimeDataXX)

observatory kinematic model setup

```
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

tma = CURRENT_TMA_DEFAULT
tma["settle_time"] = expected_wait_settle
observatory.setup_telescope(**tma)

#Set up camera with band changetime
observatory.setup_camera(band_changetime=120, readtime=3.07)

#Remove close-loop optics iterations
observatory.observatory.setup_optics(cl_delay=[0.0, 0.0], cl_altlimit=[0.0, 9.0, 90.0])
```

Anomalous Overhead Function for scatter

Add run_sim anomalous overhead 
```lsst_survey_sim.SlewScatter()``` or rubin_sim.sim_archive.AnomalousOverhead

============



