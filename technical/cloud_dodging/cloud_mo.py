from rubin_scheduler.scheduler.model_observatory import ModelObservatory
from rubin_scheduler.utils import mjd2dayobs
from rubin_scheduler.site_models import CloudMap
import os
import io
from lsst.resources import ResourcePath

import numpy as np
import healpy as hp
from astropy.time import Time

from rubin_nights import connections
from rubin_nights import lfa_data
import rubin_nights.dayobs_utils as rn_dayobs
import h5py

os.environ["S3_ENDPOINT_URL"] = "https://s3dfrgw.slac.stanford.edu/"
os.environ["LSST_DISABLE_BUCKET_VALIDATION"] = "1"


class CloudsFromDream(object):
    """Load cloud extinction from DREAM

    Parameters
    ----------
    time_limit : `float`
        How far in the past to include cloud maps.
        Default 20 (minutes).
    """
    def __init__(self, time_limit=20.0):
        self.day_obs = -1
        self.endpoints = connections.get_clients()
        self.time_limit = time_limit / 60 / 24.  # minutes to days

    def load_data(self, mjd):
        """Load the data for a given mjd
        """
        day_obs = mjd2dayobs(mjd)

        if day_obs != self.day_obs:

            sunset, sunrise = rn_dayobs.day_obs_sunset_sunrise(day_obs)
            dd = self.endpoints['efd'].select_time_series("lsst.sal.DREAM.logevent_largeFileObjectAvailable", ["url"], sunset, sunrise)
            dd = dd.query("url.str.contains('zps') and url.str.contains('hdf5')")
            mjds = []
            clouds_cleaned = []

            for i, row in dd.iterrows(): 
                uri = row.url
                uri = lfa_data.usdf_lfa(uri, bucket="s3://lfa@")
                resource = ResourcePath(uri)
                h5f = h5py.File(io.BytesIO(resource.read()), mode="r")
                dream = {}
                dream['time'] = row.name
                for k in list(h5f.keys()):
                    dream[k] = h5f[k][::]
                    dream[k] = hp.reorder(dream[k], n2r=True)
                mjds.append(Time(dream["time"]).mjd)
                try:
                    clouds_cleaned.append(dream["clouds_cleaned"])
                except:
                    import pdb ; pdb.set_trace()
            self.mjds = np.array(mjds)
            self.clouds_cleaned = np.vstack(clouds_cleaned)
            self.day_obs = day_obs

    def __call__(self, mjd):
        """Return CloudMap object populated with DREAM data
        """

        self.load_data(mjd)

        in_time_indx = np.where((self.mjds > (mjd - self.time_limit) & (self.mjds <= mjd)))[0]
        result = CloudMap()
        for indx in in_time_indx:
            result.add_frame(self.clouds_cleaned[indx], self.mjds[indx])

        return result


class CloudyModelObservarory(ModelObservatory):
    """Model Observatory that fills in cloud extinction with DREAM data
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.dream_clouds = CloudsFromDream()
        self.cloud_maps = self.dream_clouds(self.mjd)

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = self.almanac.mjd_indx(value)
        self.night = np.floor(self.mjd - self.mjd_start).astype(int)
        self.cloud_maps = self.dream_clouds(self.mjd)
