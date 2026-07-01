from pathlib import Path

from data_prep_plot.simulation_data_prep import (
    load_summary, 
    load_ph,
    gather_success_info, 
    get_success_fname,
)

# Stopping criteria
search_type = "MAP"
pmax_th = 0.95
pmax_diff_th = 1e-5
simulation_type = "sweep"  # "simple", "beam_move_res", "sweep", "mismatch", or "beam_dep"


# Path to save output csv
path_main = Path("/Volumes/ssd_2tb_1/infotaxis_simu_2026")
path_summary_ph = path_main / "20251113_pfa_sweep_MAP_summary"
path_csv = path_main / "20260608_pfa_sweep_MAP_summary_compiled_pmax095_pmaxDiff1e-5"
if not path_csv.exists():
    path_csv.mkdir(parents=True, exist_ok=True)


# Parameters to loop through
cr_all = [5, 10]
br_all = [1, 2]
pfa_all = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
pm_all = [0]


for cr in cr_all:
    for br in br_all:
        for pm in pm_all:
            for pfa in pfa_all:
                print("------------------------------------------------")
                print(f"cr={cr}, br={br}, pm={pm}, pfa={pfa}")

                # Load summary and ph details
                df_y = load_summary(
                    search_type=search_type,
                    cr=cr, br=br, 
                    path_data=path_summary_ph,
                    simulation_type=simulation_type,
                    pm_agent=pm,
                    pfa=pfa
                )

                # Get num_pings when p_max crosses threshold
                ds_y = load_ph(
                    search_type=search_type, 
                    cr=cr, br=br,
                    path_data=path_summary_ph,
                    simulation_type=simulation_type,
                    pm_agent=pm,
                    pfa=pfa
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
                    pm_agent=pm,
                    pfa=pfa,
                    simulation_type=simulation_type,
                    pmax_th=pmax_th, pmax_diff_th=pmax_diff_th,
                )
                df_y.to_csv(path_csv / csv_fname)

                print("save refined info to:")
                print(f" - {csv_fname}")
