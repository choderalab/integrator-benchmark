import os

FIGURE_PATH = os.path.join(os.path.dirname(__file__), '../figures/')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')

def create_path_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_path_if_necessary(FIGURE_PATH)
create_path_if_necessary(DATA_PATH)

from simtk import unit
simulation_parameters = {"temperature": 298 * unit.kelvin,
                            "pressure": 1 * unit.atmosphere,
                         "tolerance": 1.0 / (10**8),
                         }