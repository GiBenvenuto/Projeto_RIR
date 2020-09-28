import numpy as np
from fundus_data import FundusDataHandler
from config import get_config


def teste():
    config = get_config(True)
    dh = FundusDataHandler(True, config.im_size, config.db_size, 0)



if __name__ == "__main__":
    teste()