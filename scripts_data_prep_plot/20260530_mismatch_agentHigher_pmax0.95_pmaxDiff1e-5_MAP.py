from pathlib import Path

import numpy as np

from simulation_data_prep import (
    load_summary, 
    load_ph,
    gather_success_info, 
    get_success_fname,
)

# Stopping criteria
search_type = "MAP"
pmax_th = 0.95
pmax_diff_th = 1e-5
simulation_type = "mismatch"  # "simple", "mismatch", or "beam_dep"


# Path to save output csv
path_main = Path("/Volumes/ssd_2tb_1/infotaxis_simu_2026")
path_summary_ph = path_main / "20260524_MAP_mismatch_agentHigher_summary"
path_csv = path_main / "20260530_MAP_mismatch_agentHigher_summary_compiled_pmax095_pmaxDiff1e-5"
if not path_csv.exists():
    path_csv.mkdir(parents=True, exist_ok=True)


# Parameters to loop through
cr_all = [5, 10]
br_all = [1, 2]
pm_agent_all = [0.005, 0.01, 0.02, 0.05]
pm_truth_all = [0.001]


for cr in cr_all:
    for br in br_all:
        for pm_agent in pm_agent_all:
            for idx, pm_truth in enumerate(pm_truth_all):
                print("------------------------------------------------")
                print(f"cr={cr}, br={br}, pm_agent={pm_agent}, pm_truth={pm_truth}")

                # Load summary and ph details
                df_y = load_summary(
                    search_type=search_type,
                    cr=cr, br=br,
                    path_data=path_summary_ph,
                    simulation_type=simulation_type,
                    pm_agent=pm_agent, pm_truth=pm_truth
                )

                # Get num_pings when p_max crosses threshold
                ds_y = load_ph(
                    search_type=search_type, 
                    cr=cr, br=br, 
                    path_data=path_summary_ph,
                    simulation_type=simulation_type,
                    pm_agent=pm_agent, pm_truth=pm_truth
                )

                # Make df_y index 1-based to match with ds_y "run" dimension
                df_y.index = df_y.index + 1

                # Gather success info based on pmax_th and pmax_diff_th
                df_y = gather_success_info(
                    df_y=df_y,
                    ds_y=ds_y,
                    pmax_th=pmax_th,
                    pmax_diff_th=pmax_diff_th
                )

                # Determine output filename
                csv_fname = get_success_fname(
                    search_type=search_type,
                    cr=cr, br=br,
                    pm_agent=pm_agent, pm_truth=pm_truth,
                    simulation_type=simulation_type,
                    pmax_th=pmax_th, pmax_diff_th=pmax_diff_th,
                )
                df_y.to_csv(path_csv / csv_fname)

                print("save refined info to:")
                print(f" - {csv_fname}")
