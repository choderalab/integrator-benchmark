import os

FIGURE_PATH = os.path.join(os.path.dirname(__file__), '../figures/')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')

def create_path_if_necessary(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_path_if_necessary(FIGURE_PATH)
create_path_if_necessary(DATA_PATH)
