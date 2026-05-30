trying out profiling code:

python -m cProfile -o output_v5.3.prof baseline.py --survey_length 90 --verbose

Completed 33136 observations
ran in 18 min = 0.3 hours
Writing results to  baseline_v5.3.0_0yrs.db

conda install -c conda-forge snakeviz


now running the v5.0 (with some patch in for ToO and Roman)
Completed 35509 observations
ran in 13 min = 0.2 hours
Writing results to  baseline_v5.0.1_0yrs.db


----

trying again after adding more ignore

python -m cProfile -o output_v5.3.1.prof baseline.py --survey_length 90 --verbose

progress = 99.98%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 33136 observations
ran in 13 min = 0.2 hours
Writing results to  baseline_v5.3.1_0yrs.db


then just 
snakeviz output_v5.3.1.prof


# Trying with array matching on ignore_obs

progress = 99.99%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 32500 observations
ran in 16 min = 0.3 hours
Writing results to  baseline_v5.3.1_0yrs.db

hmm, something changed. That's a shame:  u/yoachim/ignore_speedup

try again on main:
progress = 99.98%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 33136 observations
ran in 13 min = 0.2 hours
Writing results to  baseline_v5.3.1_0yrs.db

-- ok that worked


on new branch:
python baseline.py --survey_length 10 --verbose

progress = 99.60%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 3446 observations
ran in 1 min = 0.0 hours
Writing results to  baseline_v5.3.1_0yrs.db


on main:

progress = 99.61%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 3384 observations
ran in 1 min = 0.0 hours
Writing results to  baseline_v5.3.1_0yrs.db

wtf? why diff?

----

python -m cProfile -o output_v5.3.1.prof ddf_ig.py --survey_length 90 --verbose


progress = 99.98%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 33136 observations
ran in 12 min = 0.2 hours
Writing results to  ddf_ig_v5.3.0_0yrs.db

ok, now faster than the 5.0 runs!

----

oh boy, now with cached mask 
python -m cProfile -o output_v5.3.1.prof ddf_ig.py --survey_length 90 --verbose

progress = 99.98%Skipped 0 observations
Flushed 0 observations from queue for being stale
Completed 33136 observations
ran in 8 min = 0.1 hours
Writing results to  ddf_ig_v5.3.0_0yrs.db

Booom! 2.25x faster! Now we're cooking with gas again




