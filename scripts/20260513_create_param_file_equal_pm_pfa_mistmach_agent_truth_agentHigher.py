from pathlib import Path
from datetime import date

import pandas as pd

beam_radius_all = [1, 2]
canvas_radius_all = [10]
# pm_agent_all = [0.005, 0.01]  # 1st set of run 2026/05/13
# pm_agent_all = [0.02]  # 2nd set of run 2026/05/13
# pm_agent_all = [0.05]  # 3nd set of run 2026/05/13
pm_agent_all = [0.005, 0.01, 0.02, 0.05]  # run 2026/05/24
pm_truth_all = [0.001]

df_cr = pd.DataFrame(canvas_radius_all, columns=["canvas_radius"], dtype=int)
df_br = pd.DataFrame(beam_radius_all, columns=["beam_radius"], dtype=int)
df_pm_agent = pd.DataFrame(pm_agent_all, columns=["pm_agent"])
df_pm_truth = pd.DataFrame(pm_truth_all, columns=["pm_truth"])

df_comb = (
    df_cr.merge(df_br, how="cross")
    .merge(df_pm_agent, how="cross")
    .merge(df_pm_truth, how="cross")
)
df_comb["pfa_agent"] = df_comb["pm_agent"]
df_comb["pfa_truth"] = df_comb["pm_truth"]
df_comb.index = df_comb.index +1

path_param_file = Path("/Users/wujung/code_git/infotaxis/run_param_files_2026")
param_fname = (
    f"{date.today().strftime("%Y%m%d")}"
    + "_br" + "-".join([str(b) for b in beam_radius_all])
    + "_cr" + "-".join([str(c) for c in canvas_radius_all])
    + "_equal_pm_pfa_mismatch_agent_truth.csv"
)
df_comb.to_csv(path_param_file / param_fname)