import argparse
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer

from lib.misc import load_data
from lib.preprocessing import BaselineCorrector, RangeLimiter, SavGolFilter

# Prepare logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler_c = logging.StreamHandler()
handler_f = logging.FileHandler("./log/03_preprocess_data.log")

handler_c.setLevel(logging.DEBUG)
handler_f.setLevel(logging.DEBUG)

format_c = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
format_f = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

handler_c.setFormatter(format_c)
handler_f.setFormatter(format_f)

logger.addHandler(handler_c)
logger.addHandler(handler_f)


def preprocess(data):
    X = data.loc[:, data.columns != "label"]
    wns = np.asarray(X.columns.astype(float))
    X = np.asarray(X)
    y = np.asarray(data.label)

    # Subtract baseline
    X = BaselineCorrector().fit_transform(X)

    # Reduce spectral range
    rl = RangeLimiter(lim=(450, 1670), reference=wns)
    X = rl.fit_transform(X)
    wns_reduced = wns[rl.lim_[0]:rl.lim_[1]]

    # Smooth spetra
    X = SavGolFilter().fit_transform(X)

    # Normalize intensity
    X = Normalizer().fit_transform(X)

    # Combine data back to Dataframe
    data_prep = pd.DataFrame(X, columns=wns_reduced)
    data_prep.insert(0, "label", y)

    return data_prep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a single csv file from individual Raman spectra."
    )

    parser.add_argument("-f", "--file", metavar="PATH", type=str, nargs=1, action="store",
                        help=".csv file containing the spectral data.", required=True)
    parser.add_argument("-o", "--out", metavar="PATH", type=str, nargs=1, action="store",
                        help="Path for the output file.", required=True)
    parser.add_argument("-l", "--limits", metavar=("LOW", "HIGH"), type=float, nargs=2, action="store",
                        help="Limits for reducing the spectral range.", required=False, default=(None, None))

    # TODO: Add arguments for other preprocessing steps

    logger.info("Parsing arguments")
    args = parser.parse_args()
    for arg, val in vars(args).items():
        logger.debug(f"Received argument {arg} with value {val}")

    return args


if __name__ == "__main__":
    logger.info("Starting data preprocessing")
    args = parse_args()

    path_in = Path(args.file[0])
    path_out = Path(args.out[0])

    filename = path_in.stem

    if path_out.is_dir:
        path_out = path_out / (filename + "_preprocessed.csv")

    logger.info(f"Loading data from {path_in}")
    data = load_data(path_in)
    logger.info("Finished loading data")

    data_prep = preprocess(data)

    logger.info("Preprocessing complete")
    logger.info(f"Saving preprocessed data to {path_out}")
    data_prep.to_csv(path_out, index=False)
