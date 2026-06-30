from pathlib import Path
from datetime import date

import pandas as pd

beam_radius_all = [1,2]
canvas_radius_all = [10]
pm_all = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
pfa_all = [0]

df_cr = pd.DataFrame(canvas_radius_all, columns=["canvas_radius"], dtype=int)
df_br = pd.DataFrame(beam_radius_all, columns=["beam_radius"], dtype=int)
df_pm = pd.DataFrame(pm_all, columns=["pm"])
df_pfa = pd.DataFrame(pfa_all, columns=["pfa"])
df_comb = (
    df_cr.merge(df_br, how="cross")
    .merge(df_pm, how="cross")
    .merge(df_pfa, how="cross")
)
df_comb.index = df_comb.index +1

path_param_file = Path("/Users/wujung/code_git/infotaxis/run_batch_results/param_files")
param_fname = (
    f"{date.today().strftime("%Y%m%d")}"
    + "_br" + "-".join([str(b) for b in beam_radius_all])
    + "_cr" + "-".join([str(c) for c in canvas_radius_all])
    + "_pfa0.csv"
)
df_comb.to_csv(path_param_file / param_fname)