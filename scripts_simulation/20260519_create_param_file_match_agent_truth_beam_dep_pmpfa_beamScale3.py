from pathlib import Path
from datetime import date

import pandas as pd

# # cr=5, br=1
# beam_radius_all = [1]
# canvas_radius_all = [5]
# pm_agent_all = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
# beam_sigma_r_all = [0.5]
# beam_scale_all = [3]

# # cr=5, br=2
# beam_radius_all = [2]
# canvas_radius_all = [5]
# pm_agent_all = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
# beam_sigma_r_all = [1]
# beam_scale_all = [3]


# # cr=10, br=1
# beam_radius_all = [1]
# canvas_radius_all = [10]
# pm_agent_all = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
# beam_sigma_r_all = [0.5]
# beam_scale_all = [3]

# cr=10, br=2
beam_radius_all = [2]
canvas_radius_all = [10]
pm_agent_all = [0.001, 0.002, 0.003, 0.004, 0.005, 0.01]
beam_sigma_r_all = [1]
beam_scale_all = [3]



df_cr = pd.DataFrame(canvas_radius_all, columns=["canvas_radius"], dtype=int)
df_br = pd.DataFrame(beam_radius_all, columns=["beam_radius"], dtype=int)
df_pm_agent = pd.DataFrame(pm_agent_all, columns=["pm_agent"])
df_beam_sigma_r = pd.DataFrame(beam_sigma_r_all, columns=["beam_sigma_r"])
df_beam_scale = pd.DataFrame(beam_scale_all, columns=["beam_scale"])

df_comb = (
    df_cr.merge(df_br, how="cross")
    .merge(df_pm_agent, how="cross")
    .merge(df_beam_sigma_r, how="cross")
    .merge(df_beam_scale, how="cross")
)

# matched agent and truth
df_comb["pm_truth"] = df_comb["pm_agent"]

# pm=pfa
df_comb["pfa_agent"] = df_comb["pm_agent"]
df_comb["pfa_truth"] = df_comb["pm_truth"]

df_comb.index = df_comb.index +1

path_param_file = Path("/Users/wujung/code_git/infotaxis/run_param_files_2026")
param_fname = (
    f"{date.today().strftime("%Y%m%d")}"
    + "_br" + "-".join([str(b) for b in beam_radius_all])
    + "_cr" + "-".join([str(c) for c in canvas_radius_all])
    + "_equal_pm_pfa_match_agent_truth_beam_dep_pmpfa"
    + f"_beamSigmaR{beam_sigma_r_all[0]}"
    + f"_beamScale{beam_scale_all[0]}"
    + ".csv"
)
df_comb.to_csv(path_param_file / param_fname)
