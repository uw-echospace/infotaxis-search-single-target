from pathlib import Path

from simulation_data_prep import assemble_ph, assemble_summary


path_main = Path("/Volumes/ssd_2tb_1/infotaxis_simu_2026")

path_data = path_main / "20251118"

path_save = path_main / "20251118_summary"
if not path_save.exists():
    path_save.mkdir(parents=True, exist_ok=True)

search_type = "infotaxis"
simulation_type = "beam_move_res"
dates_list = ["20251118", "20251119"]

cr_all = [5, 10]
br_all = [1, 2]
bmr_all = [2, 3]
pm_all = [0.001, 0.01, 0.02, 0.05]


for cr in cr_all:
    for br in br_all:
        for bmr in bmr_all:
            for pm in pm_all:
                print(f"cr={cr}, br={br}, bmr={bmr}, pm={pm:.0e}")

                ds = assemble_ph(
                    search_type, cr=cr, br=br, bmr=bmr,
                    pm_agent=pm,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
                df = assemble_summary(
                    search_type, cr=cr, br=br, bmr=bmr,
                    pm_agent=pm,
                    dates_list=dates_list,
                    path_data=path_data,
                    path_save=path_save,
                    simulation_type=simulation_type
                )
                
