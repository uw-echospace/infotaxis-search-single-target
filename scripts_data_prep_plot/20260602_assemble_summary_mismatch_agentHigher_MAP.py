from pathlib import Path

from data_prep_plot.simulation_data_prep import assemble_ph, assemble_summary


path_main = Path("/Volumes/ssd_2tb_1/infotaxis_simu_2026")

path_data = path_main / "20260524_MAP_mismatch_agentHigher"

path_save = path_main / "20260524_MAP_mismatch_agentHigher_summary"
if not path_save.exists():
    path_save.mkdir(parents=True, exist_ok=True)

search_type = "MAP"
simulation_type = "mismatch"
dates_list = ["20260524"]

cr_all = [5, 10]
br_all = [1, 2]
pm_truth_all = [0.001]
pm_agent_all = [0.005, 0.01, 0.02, 0.05]


for cr in cr_all:
    for br in br_all:
        for pm_agent in pm_agent_all:
            for pm_truth in pm_truth_all:

                print(f"cr={cr}, br={br}, pm_agent={pm_agent:.0e}, pm_truth={pm_truth:.0e}")

                ds = assemble_ph(
                    search_type, cr=cr, br=br,
                    pm_agent=pm_agent, pm_truth=pm_truth,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
                df = assemble_summary(
                    search_type, cr=cr, br=br,
                    pm_agent=pm_agent, pm_truth=pm_truth,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
