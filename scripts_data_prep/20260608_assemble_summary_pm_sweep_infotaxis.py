from pathlib import Path

from data_prep_plot.simulation_data_prep import assemble_ph, assemble_summary


path_main = Path("/Volumes/ssd_2tb_1/infotaxis_simu_2026")

path_data = path_main / "20251113"

path_save = path_main / "20251113_pm_sweep_infotaxis"
if not path_save.exists():
    path_save.mkdir(parents=True, exist_ok=True)

search_type = "infotaxis"
simulation_type = "sweep"
dates_list = ["20251113", "20251114"]

cr_all = [5, 10]
br_all = [1, 2]
pm_all = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
pfa_all = [0]


for cr in cr_all:
    for br in br_all:
        for pm in pm_all:
            for pfa in pfa_all:

                print(f"cr={cr}, br={br}, pm={pm:.0e}, pfa={pfa:.0e}")

                ds = assemble_ph(
                    search_type, cr=cr, br=br,
                    pm_agent=pm,
                    pfa=pfa,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
                df = assemble_summary(
                    search_type, cr=cr, br=br,
                    pm_agent=pm,
                    pfa=pfa,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
