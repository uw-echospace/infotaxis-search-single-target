# Callable script to run 1 infotaxis search

from pathlib import Path
import sys
import time
import logging
from datetime import date, datetime, timezone
import json
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from infotaxis import one_target, hex_ops



parser = argparse.ArgumentParser(
    description="Run infotaxis or MAP search."
)
# below needed regardless of if using a param file
parser.add_argument(
    "-s",
    "--search-type",
    type=str,
    default="infotaxis",
    help="search-type can be 'infotaxis' or 'MAP'"
)
parser.add_argument(
    "-n",
    "--run-number",
    type=int,
    default=1,
    help="run repetition number"
)
parser.add_argument(
    "-p",
    "--param-file",
    type=str,
    default="",
    #default="/Users/wujung/code_git/infotaxis/run_batch_results/param_files/20251019_br1-2_cr5_equal_pm_pfa.csv",
    help="path to parameter file"
)
parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    default=Path.cwd(),
    #default="/Users/wujung/code_git/infotaxis/run_batch_results/20251027_test2",
    help="output path is the directory where all run results will be saved to"
)
parser.add_argument(
    "-t",
    "--target-cube",
    type=int,
    nargs=3,
    default=[],
    #default=[1, -2, 1],
    help="target location in cube coordinate"
)
parser.add_argument(
    "-m",
    "--beam-movement-radius",
    type=int,
    default=-1,
    help="Neighbor radius to restrict beam movement (default: -1, meaning no restriction)"
)

# below are needed if not using a param file
parser.add_argument(
    "-r",
    "--random-seed",
    type=int,
    default=-1,
    help="random seed can be any integer"
)
parser.add_argument(
    "-c",
    "--canvas-radius",
    type=int,
    default=-1,
    help="radius of the search space"
)
parser.add_argument(
    "-b",
    "--beam-radius",
    type=int,
    default=-1,
    help="radius of the beam footprint"
)
parser.add_argument(
    "--pm",
    type=float,
    default=-1,
    help="probability of miss"
)
parser.add_argument(
    "--pfa",
    type=float,
    default=-1,
    help="probability of false alarm"
)



# Print inputs
args = parser.parse_args()
print("Arguments:")
for arg, value in vars(args).items():
    if not (value == "-1" or value == -1):
        print(f" - {arg}: {value}")
print("\n\n")

# Assemble simulation param dict
# This will be saved in a JSON file
today = date.today()


# Load params
search_type = args.search_type
run_number = args.run_number
path_main = Path(args.output_path)
target_cube = tuple(args.target_cube)


# Loop through all combinations in the param file
df_param = pd.read_csv(args.param_file, index_col=0)

for idx, row in df_param.iterrows():

    print(f"-----------------------------------------------")
    print(f"Parameter combination index: {idx}")

    # Get params from row
    beam_radius = int(row["beam_radius"])
    pm = float(row["pm"])
    pfa = float(row["pfa"])
    canvas_radius = int(row["canvas_radius"])

    # Generate a random target location
    canvas_cube = hex_ops.cube_within_radius((0,0,0), canvas_radius) # canvas in cube coorindate
    target_cube = canvas_cube[np.random.randint(0, canvas_cube.shape[0])].tolist()

    # Assemble full parameter set
    simu_params = {
        "search_rule": search_type,
        "param_echo": {  # pm/pfa const indexed by beam_radius
            str(beam_radius): {
                "pm_const": pm,
                "pfa_const": pfa,
            },
        },
        "target_cube": target_cube,
        "canvas_radius": canvas_radius,
        "pmax_repeat_N": 3,
        "max_ping_num": 500,
        "pmax_diff_threshold": 1e-6,
        "random_seed": run_number,
        "beam_movement_radius": args.beam_movement_radius,
        "script_name": Path(__file__).name  # current script name
    }
    print("Simulation params:")
    for k, v in simu_params.items():
        print(f" - {k}: {v}")
    print("\n")
    

    # Set spatial search constraints
    if simu_params["beam_movement_radius"] == -1:
        search_dict = None
    else:
        search_dict = {'neighbor_radius': simu_params["beam_movement_radius"]}

    # Assemble output path
    param_echo_str = []
    for k,v in simu_params["param_echo"].items():
        param_echo_str.append(
            f"_beamRadius{k}_pm{v["pm_const"]:.0e}_pfa{v["pfa_const"]:.0e}"
        )
    path_save = (
        f"{today.strftime("%Y%m%d")}_{simu_params["search_rule"]}_pmaxRepeat"
        + f"_canvasRadius{simu_params["canvas_radius"]}"
        + f"_beamMovementRadius{simu_params["beam_movement_radius"]}"
        + "".join(param_echo_str)
    )

    # Set output path
    path_save = path_main / path_save
    if not path_save.exists():
        path_save.mkdir(parents=True, exist_ok=True)


    # Stopping condition
    stopping_criteria = {
        "pmax_repeat_N": simu_params["pmax_repeat_N"],
        "max_ping_num": simu_params["max_ping_num"],
        "pmax_diff_threshold": simu_params["pmax_diff_threshold"],
    }

    # Set the random seed for numpy
    np.random.seed(simu_params["random_seed"])


    # Run search
    print(f"Start rep #{run_number:04d}")
    print(f"Start timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Record rep start time
    rep_time_start_perf = time.perf_counter()
    rep_time_start_proc = time.process_time()

    oth = one_target.OneTargetHex(
        search_rule=simu_params["search_rule"],  # search rule: "infotaxis", "MAP_future", "MAP"
        canvas_radius=simu_params["canvas_radius"],
        beam_radius=list(simu_params["param_echo"].keys()),
        target_cube=simu_params["target_cube"],
        # aim_start_cube=(0, 0, 0),  # non-random start
        param_animal= simu_params["param_echo"],
        param_echo= simu_params["param_echo"],
        search_dict=search_dict
    )

    continue_to_ping = True
    ping_counter = 1

    while True:
        print(f"Ping num {ping_counter}")

        # Generate an echo
        oth.get_echo()

        # Update the dk map according to echo outcome
        if oth.echo_value:
            oth.update_X1()
        else:
            oth.update_X0()

        # Update h after updating the map
        oth.get_h_actual()

        # Get another round of h_est and determine the next beam
        if simu_params["search_rule"] in ["infotaxis", "MAP_future"]:
            oth.get_est_ph()
        oth.get_next_beam()

        # Save all updates on params
        oth.update_record()

        # Record rep end time
        rep_time_end_perf = time.perf_counter()
        rep_time_end_proc = time.process_time()

        # Decide if to continue to ping
        if oth.check_stopping(criteria=stopping_criteria):

            num_ping = len(oth.h_actual_all)-1
            num_grid_left = int((oth.d_all[-1].max() == oth.d_all[-1]).sum())
            p_max = np.max(oth.d_all, axis=1)
            
            print("")
            print(f"num_grid_left: {num_grid_left}")
            print(f"Diff of p_max: {np.diff(p_max)}")
            print("")

            print(f"End timestamp: {datetime.now(timezone.utc).isoformat()}")
            print(f"------- finish search #{run_number:04d} at ping #{num_ping:03d}")
            print(f"----------------------------------------------- \n\n\n\n")
            break
        else:
            ping_counter += 1

    
    # Save simulation results
    fname_prefix = f"{simu_params["search_rule"]}_{run_number:04d}"

    oth.save_to_nc(
        path_save / f"{fname_prefix}.nc",
        time_spent_perf=rep_time_end_perf-rep_time_start_perf,
        time_spent_proc=rep_time_end_proc-rep_time_start_proc,
    )

    # Save simulation params to json
    with open(path_save / f"{fname_prefix}_config.json", "w") as f:
        json.dump(simu_params, f, indent=4)

    # Save summary info to csv
    num_pings = len(oth.h_actual_all)
    num_grid_left = (oth.d_all[-1].max() == oth.d_all[-1]).sum()
    time_perf = rep_time_end_perf-rep_time_start_perf
    time_proc = rep_time_end_proc-rep_time_start_proc
    files_simu = str(path_save / f"{fname_prefix}.nc")
    df_summary = pd.DataFrame(
        [num_pings, num_grid_left, time_perf, time_proc, files_simu],
        index=["num_pings", "num_grid_left", "time_perf", "time_proc", "files_simu"]
    ).T
    df_summary.to_csv(path_save / f"{fname_prefix}_summary.csv")
    
    # Save detailed h_actual and p_max across pings to csv
    df_ph = pd.DataFrame(
        [oth.h_actual_all, np.max(oth.d_all, axis=1)],
        index=["h_actual_all", "p_max_all"]
    ).T
    df_ph.to_csv(path_save / f"{fname_prefix}_ph.csv")
