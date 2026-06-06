from mad.configs.physics_cfg import MILES_TO_METERS

# See https://en.wikipedia.org/wiki/Solid_State_Phased_Array_Radar_System
BMEWS = {
    "name": "EWR",
    "max_range_m": 3000 * MILES_TO_METERS,  # 4800 km
    "voxel_size": 500_000.0,  # m, larger voxel size for early warning radars
}

# See https://en.wikipedia.org/wiki/AN/SPY-1
AN_SPY_1 = {
    "name": "Spy-1",
    "max_range_m": 370_000,
    "voxel_size": 100_000.0,  # m, smaller voxel size for tracking radars
}
