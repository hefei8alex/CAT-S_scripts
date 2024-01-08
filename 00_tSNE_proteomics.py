# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from typing import Tuple

# set parameters
INPUT_PATH = "./example_data/all_cell_lines.csv"          # .csv
OUTPUT_PATH = "./example_data/all_cell_lines_tsne.csv"    # .csv
DATA_COLS = ["293T", "HeLa", "K562" ]
ID_COL = "UniProt ID"
GROUP_COL = "group"

RESCALE = True
PLOT = True
EXPORT = True


def load_data(path: str=INPUT_PATH) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    data = np.array(df[DATA_COLS])

    return df, data


def standard_scale(data: np.ndarray) -> np.array:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled


def tSNE_analysis(df: pd.DataFrame, data: np.ndarray, n_components=2, perplexity=20) -> pd.DataFrame:
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    data_tSNE = tsne.fit_transform(data)
    data_tSNE = pd.DataFrame(data_tSNE, columns=["tSNE-1", "tSNE-2"])
    df_tSNE = pd.concat([df[ID_COL], data_tSNE, df[GROUP_COL]], axis=1)

    return df_tSNE


def plot_tSNE_preview(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    colors = ['red', 'gray', 'purple', 'blue', 'orange', 'green', 'magenta', 'gold', 'aqua', 'purple', 'tan']
    groups = df[GROUP_COL].unique()
    for i, group in enumerate(groups):
        df_i = df[df[GROUP_COL] == group]
        plt.scatter(df_i["tSNE-1"], df_i["tSNE-2"], c=colors[i])
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.legend(groups)
    plt.show()


def export_to_file(df: pd.DataFrame, path: str=OUTPUT_PATH):
    import os
    df.to_csv(path, index=False)
    print("Exported to %s." % os.path.abspath(path))


if __name__ == '__main__':

    df, data = load_data()

    if RESCALE:
        data = standard_scale(data)

    df_tSNE = tSNE_analysis(df, data)

    if PLOT:
        plot_tSNE_preview(df_tSNE)

    if EXPORT:
        export_to_file(df_tSNE)




