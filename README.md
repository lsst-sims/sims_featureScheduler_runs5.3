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
v5.3 - start date June 15, 2026 (old scheduled downtime??)


v5.0 - 22 weeks (2 weeks per year) scheduled downtime, UnscheduledDowntimeMoreY1Data unscheduled downtime (70% to 99% over Y1)

v5.3 - new downtime
New planned downtime - 46 weeks of block scheduled downtime + 18 weeks of single night engineering time
Unscheduled downtime - 80% rising to 97% after Y2 for short periods of downtime + old 21 weeks of unscheduled downtime

v5.0 - slew performance 40% TMA + 3s settle
v5.3 - slew performance 20% TMA + 3s settle (+scatter?), moving to 20% TMA + 1.5s settle after Y1 + reduced scatter


Requested sims: 
* new baseline configuration + old perf
* new baseline configuration + new perf
* rolling after year 2 + new perf
* desi + new perf
* template tier with shorter exposure 20s + new perf
