Working off of comp_survey, but modifying downtime and slew performance

Adding in the max_pair_time to observation reason on the blobs.
Adding the block size to the triplet blob section as well.


running for 11 years. See a big boost in things partly because there are 
no DDF observations in the 11th year. Now added ToOs to 11th year.


going up to 5.3.1:
* Changes for speeding up sims by moving some calcs to `Conditions` object
* Have scripted surveys ignore most things

nominally those should not have changed anything, but looks like 5.3.0 and 5.3.1 are not identical. But really really close. 


going up to 5.3.2:
* Trying out flushing the queue when new ToO alerts come in

going up to 5.3.3:
* bug fix on dynamic pairs
* lower scheduled_respect kwarg to 15 so we get shorter pairs.
* might have been a slight change in downtime swapping back to main from a branch of lsst_survey_sim

going to 5.3.4:
* Making all bands available to greedy survey. Prevents a handful of filter changes, but seems to be positive beyond that.

going to 5.3.5:
* more dynamic pair fixes
* no blue greedy in bright time
* not resetting templates per season
* fixing some lost SSO ToO events
* fix n_snaps on GW ToO


**Bleeding Edge, not yet final**
going to 5.3.6: 
* One more fixed n_snaps for GW case_A
