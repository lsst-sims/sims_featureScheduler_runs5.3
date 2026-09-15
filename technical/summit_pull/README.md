Pull down the config from https://github.com/lsst-ts/ts_fbs_utils/tree/tickets/SP-3220/python/lsst/ts/fbs/utils/maintel

and check if there are any differences to the baseline that need to propogate up


and pull fbs_config_lsst_survey from https://github.com/lsst-ts/ts_config_scheduler/tree/tickets/SP-3402/Scheduler/feature_scheduler/maintel


and pull DDF config and generation from ts_config_scheduler/Scheduler/ddf_gen (develop branch)


notes:

 * The DDF generation was using NEXP variable. So DDF scripts need to be updated or put NEXP back. DDF needs an update for EDFS ultra deep going later season anyway
* The STANDARD_MASK_DEFAULTS dict was confusing and causing errors. I think I took it out reasonably, but the values were different that the default kwarg values in `standard_mask`, so unclear which way to resolve that. 
* Should survey start date be updated? 
* Had to put template tier back in 
