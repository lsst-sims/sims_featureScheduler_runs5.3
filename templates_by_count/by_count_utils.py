import numpy as np
from rubin_scheduler.scheduler.utils import Footprint
from rubin_scheduler.scheduler.basis_functions import FootprintBasisFunction
from rubin_scheduler.utils import DEFAULT_NSIDE


class FootprintCount(Footprint):
    """Rather than normalized footprint, return the raw
    step function value at each HEALpix """

    def __init__(
        self,
        mjd_start,
        sun_ra_start=0,
        nside=DEFAULT_NSIDE,
        bands=["u", "g", "r", "i", "z", "y"],
        filters=None,
        period=365.25,
        step_func=None,
    ):
        super().__init__(
            mjd_start,
            sun_ra_start=sun_ra_start,
            nside=nside,
            bands=bands,
            period=period,
            step_func=step_func,
        )

    def _update_mjd(self, mjd, norm=False):
        if mjd != self.mjd_current:
            self.mjd_current = mjd
            t_elapsed = mjd - self.mjd_start

            norm_coverage = self.step_func(t_elapsed, self.phase)
            norm_coverage -= self.zero
            self.current_footprints = self.footprints * norm_coverage


class FootprintCountBasisFunction(FootprintBasisFunction):
    """Use a FootprintCount for when you want to specify
    explicit number of observations rather than ratios.

    Parameters
    ----------
    mulit_factor : `float`
        Factor to mulitiply footprint by. For a footprint
        with values on 1, mult_factor will set the number of visits 
        per year desired. 
    mask_below_val : `float`
        Mask where the basis function drops below this value. 
        Can be used to prevent footprint from getting too far
        ahead.
    mask_above_count : `float`
        Mask where the total number of visits exceeds.
    """
    def __init__(
        self,
        bandname="r",
        nside=DEFAULT_NSIDE,
        footprint=None,
        out_of_bounds_val=-10.0,
        filtername=None,
        seeing_fwhm_max=None,
        seeing_fill_value=100.0,
        mult_factor=3,
        mask_reward_below=-2,
        mask_above_count=7,

    ):

        super().__init__(
            bandname=bandname,
            nside=nside,
            footprint=footprint,
            out_of_bounds_val=out_of_bounds_val,
            filtername=filtername,
            seeing_fwhm_max=seeing_fwhm_max,
            seeing_fill_value=seeing_fill_value,
        )
        self.mult_factor = mult_factor
        self.mask_reward_below = mask_reward_below
        self.mask_above_count = mask_above_count

    def __call__(self, conditions, indx=None):

        desired_footprint_counts = self.footprint(conditions.mjd)[self.bandname]

        result = desired_footprint_counts*self.mult_factor - self.survey_features["N_obs"].feature

        if self.mask_reward_below is not None:
            indx = np.where(result <= self.mask_reward_below)[0]
            result[indx] = np.nan
        if self.mask_above_count is not None:
            indx = np.where(self.survey_features["N_obs"].feature >= self.mask_above_count)[0]
            result[indx] = np.nan

        return result
