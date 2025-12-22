from collections import defaultdict
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Set data path
path_main = Path("/scratch/ch153/wjl/infotaxis/runs")

# path_infotaxis_dict = {
#     "cr5": "20251025/simu_output",
#     "cr8": "20251025/simu_output",
#     "cr10": "20251025/simu_output",
# }
# path_MAP_dict = {
#     "cr5": "20251025/simu_output",
#     "cr8": "20251025/simu_output",
#     "cr10": "20251025/simu_output",
# }

path_infotaxis_dict = {
    "cr5": "20251026_revised_run/simu_output",
    "cr8": "20251026_revised_run/simu_output",
    "cr10": "20251026_revised_run/simu_output",
}
path_MAP_dict = {
    "cr5": "20251026_revised_run/simu_output",
    "cr8": "20251026_revised_run/simu_output",
    "cr10": "20251026_revised_run/simu_output",
}

# path_save = path_main / "20251026"
path_save = path_main / "20251026_revised_run_summary"
if not path_save.exists():
    path_save.mkdir(parents=True, exist_ok=True)


# Parameter combinations
cr_all = [5, 8, 10]
# cr_all = [10]
br_all = [1, 2]
# br_all = [1]
pm_all = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
# pm_all = [0.02]


def gather_run_stats(search_type, path_save):

    for cr in cr_all:
        for br in br_all:
            for pm in pm_all:

                # Initialize data container
                num_pings = [] # number of pings (including initial condition)
                num_grid_left = [] # number of grids at the end of search
                time_perf = [] # time to run the whole search (perf_counter)
                time_proc = [] # time to run the whole search (process_time)
                h_actual = [] # actual entropy across all pings
                p_max = [] # max target probability across all pings

                # Only cases when PM=PFA
                pfa = pm

                # Assemble dict key
                key = f"cr{cr}_br{br}_pm{pm:.0e}_pfa{pfa:.0e}"
                print(key)

                # Assemble paths
                if search_type == "infotaxis":
                    path_simu = path_main / path_infotaxis_dict[f"cr{cr}"]
                else:
                    path_simu = path_main / path_MAP_dict[f"cr{cr}"]
                folder_pattern = (
                    f"pmaxRepeat"
                    + f"_canvasRadius{cr}"
                    + f"_beamRadius{br}"
                    + f"_pm{pm:.0e}"
                    + f"_pfa{pfa:.0e}"
                )
                folders = list(path_simu.rglob(f"*_{search_type}_{folder_pattern}"))
                print("folder:")
                for ff in folders:
                    print(ff)

                # Find all files from all folders
                files_simu = []
                for ff in folders:
                    files_simu = files_simu + list(ff.glob("*.nc"))
                files_simu = sorted(files_simu)

                # Extract data from all runs
                print(f"Processing {len(files_simu)} files...")
                for ff in files_simu:
                    ds = xr.open_dataset(ff)
                    num_pings.append(len(ds["h_actual_all"]))
                    num_grid_left.append(
                        (ds["d_all"].isel(steps=-1).max() == ds["d_all"].isel(steps=-1)).sum().values
                    )
                    time_perf.append(ds.attrs["search_time_perf"])
                    time_proc.append(ds.attrs["search_time_proc"])
                    h_actual.append(ds["h_actual_all"].values)
                    p_max.append(np.max(ds["d_all"].values, axis=1))

                # Pad ragged list to array
                len_max = max(len(h) for h in h_actual)
                h_actual_pad = np.array([h.tolist() + [None] * (len_max - len(h)) for h in h_actual])
                p_max_pad = np.array([p.tolist() + [None] * (len_max - len(p)) for p in p_max])

                # Store extracted data into dataframe
                fname = f"{search_type}_{key}"
                print(f"Save data to folder: {path_save}")
                # Summary data
                df = pd.DataFrame(
                    [num_pings, num_grid_left, time_perf, time_proc, files_simu],
                    index=["num_pings", "num_grid_left", "time_perf", "time_proc", "files_simu"]
                ).T
                df.to_csv(path_save / f"{fname}_summary.csv")
                print(f" - summary data:          {fname}_summary.csv")
                # Detailed h_actual and p_max across pings
                ds = xr.Dataset(
                    data_vars=dict(
                        h_actual=(["run", "ping"], h_actual_pad),
                        p_max=(["run", "ping"], p_max_pad),
                    ),
                    coords=dict(
                        run=("run", range(1, 1001)),
                        ping=("ping", range(len_max)),
                    ),
                )
                ds.to_netcdf(path_save / f"{fname}_ph.nc")
                print(f" - h_actual and p_max to: {fname}_ph.nc")

                print("")


gather_run_stats(search_type="infotaxis", path_save=path_save)