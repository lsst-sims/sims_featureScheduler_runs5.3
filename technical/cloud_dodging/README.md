Try to set up a model observatory that pulls DREAM cloud data


have:
* cloud_dodge:  DREAM clouds with the queue manager set to default cloud dodging. M5Diff basis function includes cloud extinction.
* cloud_no_dodge: DREAM clouds with queue manager set to do nothing. Note standard basis functions set to not use cloud map in M5Diff basis function.
* no_clouds:  No DREAM clouds at all.

Setting greedy survey to use cloud future predicted values.


Was getting failed runs trying to go for a full year, but no Errors in the logs. So that's hard to debug. Seems to work now. (~Sept 7, 2026)



