from utils import configure_platform, load_waterbox, load_alanine
from simtk import unit

system_params = {
    "waterbox": {
        "platform" : configure_platform("OpenCL"),
        "loader": load_waterbox,
        "burn_in_length": 100,
        "n_samples": 50,
        "protocol_length": 25,
        "constrained_timestep": 2.5*unit.femtosecond,
        "unconstrained_timestep": 1.0*unit.femtosecond,
        "temperature": 298.0 * unit.kelvin,
        "collision_rate": 91 / unit.picoseconds,
    },
    "alanine": {
        "platform": configure_platform("Reference"),
        "loader": load_alanine,
        "burn_in_length": 1000,
        "n_samples": 1000,
        "protocol_length": 50,
        "constrained_timestep": 2.5*unit.femtosecond,
        "unconstrained_timestep": 2.0*unit.femtosecond,
        "temperature": 298.0 * unit.kelvin,
        "collision_rate": 91 / unit.picoseconds,
    }
}