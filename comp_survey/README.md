
Changes from v5.2

* moving deep DDF season out of year 1 (to free up dark time for templates)
* Increasing number of desired template images (inside science radius) to 6 from 4
* modifying template tier to have multiple area + hour angle thresholds
* removing 15 min pair tier
* 33 min pair tier should be able to contract if needed to respect scheduled observations and twilight time
* updates to what is recorded in scheduler_note field for pair_33s (so we can see if it dynamically changed blob size to expand/contract to fill time)

Note I've added a single SURVEY_START_DATE location, so DDF and survey are tied to the same start. Might want to make it clear if we seperate DDF start date from the rest. Actually, the ddf start should probably go inside the ddf file because of the hashing.


