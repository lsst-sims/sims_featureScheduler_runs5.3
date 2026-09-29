__all__ = ("FootprintMod1",)

from rubin_scheduler.scheduler.utils import Phase3AreaMap


class FootprintMod1(Phase3AreaMap):
    def __init__(self, dust_limit=0.11, **kwargs):
        super().__init__(dust_limit=dust_limit, **kwargs)

