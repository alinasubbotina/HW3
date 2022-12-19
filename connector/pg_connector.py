import pandas as pd
from conf.conf import logging


def get_data(link: str) -> pd.DataFrame:
    """
    This function extracts data from link
    """
    logging.info("Extracting dataframe")
    df = pd.read_csv(link)
    logging.info("Dataframe is extracted")
    return df
