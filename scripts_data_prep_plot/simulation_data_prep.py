from typing import Literal, Optional
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def assemble_summary(
    search_type: str,
    cr: int, br: int,
    dates_list: list[str],
    path_data: Path,
    path_save: Path,
    simulation_type: Literal["simple", "beam_move_res", "sweep", "mismatch", "beam_dep"],
    pm_agent: float,
    pm_truth: Optional[float]=None,
    pfa: Optional[float]=None,
    bmr: Optional[int]=None,
    beam_sigma_r: Optional[float]=None, beam_scale: Optional[float]=None,
):
    
    files = []
    for d in dates_list:
        if simulation_type == "simple":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_summary.csv"
        elif simulation_type == "beam_move_res":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamMovementRadius{bmr}_beamRadius{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_bmr{bmr}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_summary.csv"
        elif simulation_type == "sweep":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}_beamRadius{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}_summary.csv"
        elif simulation_type == "mismatch":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}"
                f"_beamRadius{br}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_summary.csv"
        elif simulation_type == "beam_dep":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}"
                f"_beamRadius{br}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}"
            )
            output_fname = (
                f"{search_type}_cr{cr}_br{br}"
                f"_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}_summary.csv"
            )
        else:
            raise ValueError(f"Invalid simulation_type: {simulation_type}")
        files += sorted(list((path_data / input_folder).glob("*_summary.csv")))

    data_list = []
    for f in files:
        data_list.append(pd.read_csv(f, index_col=0))
    df = pd.concat(data_list).reset_index(drop=True)
    
    print(f"Saved assembled summary to {path_save / output_fname}")  
    df.to_csv(path_save / output_fname)


def assemble_ph(
    search_type: str,
    cr: int, br: int,
    dates_list: list[str],
    path_data: Path,
    path_save: Path,
    simulation_type: Literal["simple", "beam_move_res", "sweep", "mismatch", "beam_dep"],
    pm_agent: float,
    pm_truth: Optional[float]=None,
    pfa: Optional[float]=None,
    bmr: Optional[int]=None,
    beam_sigma_r: Optional[float]=None, beam_scale: Optional[float]=None,
    path_length: bool=False
):
    files_ph = []
    files_dk_max_cube = []
    files_path_length = []
    for d in dates_list:
        if simulation_type == "simple":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_ph.nc"
        elif simulation_type == "beam_move_res":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamMovementRadius{bmr}_beamRadius{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_bmr{bmr}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_ph.nc"
        elif simulation_type == "sweep":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}_beamRadius{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}_ph.nc"
        elif simulation_type == "mismatch":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}"
                f"_beamRadius{br}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
            )
            output_fname = f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_ph.nc"
        elif simulation_type == "beam_dep":
            input_folder = (
                f"{d}_{search_type}_pmaxRepeat_canvasRadius{cr}"
                f"_beamRadius{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}"
                f"_beamRadius{br}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}"
            )
            output_fname = (
                f"{search_type}_cr{cr}_br{br}"
                f"_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}"
                f"_beamSigmaR{beam_sigma_r:.0e}_beamScale{beam_scale:.0e}_ph.nc"
            )
        else:
            raise ValueError(f"Invalid simulation_type: {simulation_type}")
        files_ph += sorted(list((path_data / input_folder).glob("*_ph.csv")))
        files_dk_max_cube += sorted(list((path_data / input_folder).glob("*_dk_max_cube.nc")))
        if path_length:
            files_path_length += sorted(list((path_data / input_folder).glob("*_path_length.nc")))


    # Sanity check the file number matches
    assert len(files_ph) == len(files_dk_max_cube), (
        f"Number of _ph files ({len(files_ph)}) does not match number of "
        f"_dk_max_cube files ({len(files_dk_max_cube)})"
    )
    if path_length:
        assert len(files_ph) == len(files_path_length), (
            f"Number of _ph files ({len(files_ph)}) does not match number of "
            f"_path_length files ({len(files_path_length)})"
        )

    # Assemble data into lists
    p_list = []
    h_list = []
    path_length_list = []
    target_list = []
    dk_max_cube_list = []
    for idx, (f_ph, f_dk_max_cube) in enumerate(zip(files_ph, files_dk_max_cube)):
        df = pd.read_csv(f_ph, index_col=0)
        p_list.append(df["p_max_all"])
        h_list.append(df["h_actual_all"])

        ds_out = xr.open_dataset(f_dk_max_cube)
        target_list.append(ds_out["target_cube"].values)
        dk_max_cube_list.append(ds_out["max_dk_cube"].values)

        if path_length:
            f_path_length = files_path_length[idx]                
            ds_path_length = xr.open_dataset(f_path_length)
            path_length_list.append(ds_path_length["path_length"].values)


    # If less than 1000 runs, print out missing run
    if len(h_list) < 1000:
        run_nums = []
        for f in files_ph:
            a = re.match(r"infotaxis_(\d{4})_ph.csv", f.name)
            run_nums.append(int(a.groups()[0]))
        missing_runs = set(range(1, 1001)) - set(run_nums)
        print(
            f"Missing runs for cr={cr}, br={br}, "
            f"pm_agent={pm_agent:.0e}, pm_truth={pm_truth:.0e}: {missing_runs}"
        )

    # Pad ragged list to array
    len_max = max(len(h) for h in h_list)
    h_actual_pad = np.array([h.tolist() + [np.nan] * (len_max - len(h)) for h in h_list])
    p_max_pad = np.array([p.tolist() + [np.nan] * (len_max - len(p)) for p in p_list])
    target_cube_pad = np.array(target_list)
    dk_max_cube_pad = np.array([
        np.vstack((d, np.nan * np.ones((len_max - len(d), 3)))) for d in dk_max_cube_list
    ])
    if path_length:
        path_length_pad = np.array([
            np.hstack((p, np.nan * np.ones((len_max - len(p))))) for p in path_length_list
        ])

    ds_out = xr.Dataset(
        data_vars=dict(
            h_actual=(["run", "ping"], h_actual_pad),
            p_max=(["run", "ping"], p_max_pad),
            target_cube=(["run", "cube_dim"], target_cube_pad),
            dk_max_cube=(["run", "ping", "cube_dim"], dk_max_cube_pad),
        ),
        coords=dict(
            run=("run", range(1, len(h_list) + 1)),
            ping=("ping", range(len_max)),
            cube_dim=("cube_dim", ["1", "2", "3"]),
        )
    )
    if path_length:
        ds_out["path_length"]=(["run", "ping"], path_length_pad)
    
    print(f"Saving assembled data to {path_save / output_fname}")
    ds_out.to_netcdf(path_save / output_fname)


def load_summary(
    search_type: str,
    cr: int, br: int,
    path_data: Path,
    simulation_type: Literal["simple", "beam_move_res", "sweep", "mismatch", "beam_dep"],
    pm_agent: float,
    pm_truth: Optional[float]=None,
    pfa: Optional[float]=None,
    bmr: Optional[int]=None,
    beam_sigma_r: Optional[float]=None, beam_scale: Optional[float]=None,
):
    if simulation_type == "simple":
        csv_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_summary.csv"
    elif simulation_type == "beam_move_res":
        csv_fname = f"{search_type}_cr{cr}_br{br}_bmr{bmr}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_summary.csv"
    elif simulation_type == "sweep":
        csv_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}_summary.csv"
    elif simulation_type == "mismatch":
        csv_fname = f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_summary.csv"
    elif simulation_type == "beam_dep":
        csv_fname = (
            f"{search_type}_cr{cr}_br{br}"
            f"_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}"
            f"_beam_sigma_r{beam_sigma_r:.0e}_beam_scale{beam_scale:.0e}_summary.csv"
        )
    else:
        raise ValueError(f"Invalid simulation_type: {simulation_type}")

    return pd.read_csv(path_data / csv_fname, index_col=0)


def load_ph(
    search_type: str,
    cr: int, br: int,
    path_data: Path,
    simulation_type: Literal["simple", "beam_move_res", "sweep", "mismatch", "beam_dep"],
    pm_agent: float,
    pm_truth: Optional[float]=None,
    pfa: Optional[float]=None,
    bmr: Optional[int]=None,
    beam_sigma_r: Optional[float]=None, beam_scale: Optional[float]=None,
):
    if simulation_type == "simple":
        nc_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_ph.nc"
    elif simulation_type == "beam_move_res":
        nc_fname = f"{search_type}_cr{cr}_br{br}_bmr{bmr}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}_ph.nc"
    elif simulation_type == "sweep":
        nc_fname = f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}_ph.nc"
    elif simulation_type == "mismatch":
        nc_fname = f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_ph.nc"
    elif simulation_type == "beam_dep":
        nc_fname = (
            f"{search_type}_cr{cr}_br{br}"
            f"_pmAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}"
            f"_beam_sigma_r{beam_sigma_r:.0e}_beam_scale{beam_scale:.0e}_ph.nc"
        )
    else:
        raise ValueError(f"Invalid simulation_type: {simulation_type}")

    return xr.open_dataset(path_data / nc_fname)


def get_ping_max_of_run(ds):
    ping_max_all = []
    for run in ds["run"].values:
        ping_max = ds["p_max"].sel(run=run).dropna(dim="ping")["ping"].max().values
        ping_max_all.append(int(ping_max))
    return np.array(ping_max_all)


def get_ping_cross_pmax_th(ds, pmax_th=0.95):
    ping_cross_all = []
    for run in ds["run"].values:
        ping_cross_th = ds["p_max"].sel(run=run).dropna(dim="ping").values > pmax_th
        if ping_cross_th.sum() > 0:
            ping_cross = np.argwhere(ping_cross_th).min()
        else:
            ping_cross = -1  # Use -1 to indicate no crossing
            print(f"-- run {run}: no p_max crossed the threshold of {pmax_th}.")
        ping_cross_all.append(int(ping_cross))
    return np.array(ping_cross_all)


def get_ping_pmax_diff_th(ds, pmax_diff_th=1e-5):
    criteria = {
        "pmax_repeat_N": 3,
        # "max_ping_num": max_ping_num,  # not used here
        "pmax_diff_threshold": pmax_diff_th,
    }    
    p_all = []
    for run in ds["run"].values:
        p_max = ds["p_max"].sel(run=run).dropna(dim="ping")
        for p in range(len(p_max)):
            if p - criteria["pmax_repeat_N"] < 0:
                continue
            p_max_diff = np.diff(p_max[p-criteria["pmax_repeat_N"]+1:p+1])
            if len(p_max_diff) > 1 and np.all(abs(p_max_diff) < criteria["pmax_diff_threshold"]):
                p_all.append(p)  # Append the ping index where the criterion is met
                break
            if p == len(p_max) - 1:
                p_all.append(-1)  # Use -1 to indicate criterion not met at the end of run
                print(f"-- run {run}: reached the end of p_max without meeting the pmax diff criterion.")
                break
    return np.array(p_all)


def get_success_fname(
    search_type: str,
    cr: int, br: int,
    simulation_type: Literal["simple", "beam_move_res", "sweep", "mismatch", "beam_dep"],
    pmax_th: float, pmax_diff_th: float,
    pm_agent: float,
    pm_truth: Optional[float]=None,
    pfa: Optional[float]=None,
    bmr: Optional[int]=None,
    beam_sigma_r: Optional[float]=None, beam_scale: Optional[float]=None,
):
    if simulation_type == "simple":
        return (
            f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            f"_summary_pMaxTh{pmax_th:.1e}_pMaxDiffTh{pmax_diff_th:.0e}.csv"
        )
    elif simulation_type == "beam_move_res":
        return (
            f"{search_type}_cr{cr}_br{br}_bmr{bmr}_pm{pm_agent:.0e}_pfa{pm_agent:.0e}"
            f"_summary_pMaxTh{pmax_th:.1e}_pMaxDiffTh{pmax_diff_th:.0e}.csv"
        )
    elif simulation_type == "sweep":
        return (
            f"{search_type}_cr{cr}_br{br}_pm{pm_agent:.0e}_pfa{pfa:.0e}"
            f"_summary_pMaxTh{pmax_th:.1e}_pMaxDiffTh{pmax_diff_th:.0e}.csv"
        )
    elif simulation_type == "mismatch":
        return (
            f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
            f"_summary_pMaxTh{pmax_th:.1e}_pMaxDiffTh{pmax_diff_th:.0e}.csv"
        )
    elif simulation_type == "beam_dep":
        return (
            f"{search_type}_cr{cr}_br{br}_pmAgent{pm_agent:.0e}_pfaAgent{pm_agent:.0e}_pmTruth{pm_truth:.0e}_pfaTruth{pm_truth:.0e}"
            f"_beam_sigma_r{beam_sigma_r:.0e}_beam_scale{beam_scale:.0e}"
            f"_summary_pMaxTh{pmax_th:.1e}_pMaxDiffTh{pmax_diff_th:.0e}.csv"
        )
    else:
        raise ValueError(f"Invalid simulation type: {simulation_type}")


def gather_success_info(
    df_y: pd.DataFrame,
    ds_y: xr.Dataset,
    pmax_th: float,
    pmax_diff_th: float,
    path_length: bool=False
):
    # Number of pings based on p_max threshold and p_max difference threshold
    y1 = get_ping_cross_pmax_th(ds_y, pmax_th=pmax_th)
    y2 = get_ping_pmax_diff_th(ds_y, pmax_diff_th=pmax_diff_th)
    ping_max_run = get_ping_max_of_run(ds_y)  # max ping number of each run
    df_y["num_pings_pmax_th"] = y1
    df_y["num_pings_pmax_diff"] = y2
    df_y["num_pings_run"] = ping_max_run

    # Get index to check target location
    y1_check = y1.copy()
    y2_check = y2.copy()
    # For runs that did not cross the threshold, use the last ping to check target location
    y1_check[y1_check == -1] = ping_max_run[y1_check == -1]
    y2_check[y2_check == -1] = ping_max_run[y2_check == -1]

    # Check if max dk cube is the same as target location at threshold pings
    ds_y["y1_check"] = (["run"], y1_check)
    ds_y["y2_check"] = (["run"], y2_check)
    ds_y["y_run"] = (["run"], ping_max_run)
    df_y["success_pmax_th"] = (
        (ds_y["dk_max_cube"].sel(ping=ds_y["y1_check"]) == ds_y["target_cube"])
        .sum(dim="cube_dim") == 3
    )
    df_y["success_pmax_diff"] = (
        (ds_y["dk_max_cube"].sel(ping=ds_y["y2_check"]) == ds_y["target_cube"])
        .sum(dim="cube_dim") == 3
    )
    df_y["success_run"] = (
        (ds_y["dk_max_cube"].sel(ping=ds_y["y_run"]) == ds_y["target_cube"])
        .sum(dim="cube_dim") == 3
    )

    # Set success entry to False for runs that did not cross the threshold (y1 or y2 == -1)
    df_y.loc[df_y["num_pings_pmax_th"] == -1, "success_pmax_th"] = False
    df_y.loc[df_y["num_pings_pmax_diff"] == -1, "success_pmax_diff"] = False

    # Get cumulative path length
    if path_length:
        ds_y["path_length_cumsum"] = ds_y["path_length"].cumsum(dim="ping", skipna=False)

    # Get pmax, h_actual, and cumulative path_length at threshold pings
    df_y["pmax_pmax_th"] = ds_y["p_max"].sel(ping=ds_y["y1_check"]).values
    df_y["pmax_pmax_diff"] = ds_y["p_max"].sel(ping=ds_y["y2_check"]).values
    df_y["pmax_run"] = ds_y["p_max"].sel(ping=ds_y["y_run"]).values
    df_y["h_actual_pmax_th"] = ds_y["h_actual"].sel(ping=ds_y["y1_check"]).values
    df_y["h_actual_pmax_diff"] = ds_y["h_actual"].sel(ping=ds_y["y2_check"]).values
    df_y["h_actual_run"] = ds_y["h_actual"].sel(ping=ds_y["y_run"]).values
    if path_length:
        df_y["path_length_cumsum_pmax_th"] = ds_y["path_length_cumsum"].sel(ping=ds_y["y1_check"]).values
        df_y["path_length_cumsum_pmax_diff"] = ds_y["path_length_cumsum"].sel(ping=ds_y["y2_check"]).values
        df_y["path_length_cumsum_run"] = ds_y["path_length_cumsum"].sel(ping=ds_y["y_run"]).values
    # For runs that did not cross the threshold, set pmax, h_actual, and cumulative path_length to NaN
    df_y.loc[df_y["num_pings_pmax_th"] == -1, ["pmax_pmax_th", "h_actual_pmax_th", "path_length_cumsum_pmax_th"]] = np.nan
    df_y.loc[df_y["num_pings_pmax_diff"] == -1, ["pmax_pmax_diff", "h_actual_pmax_diff", "path_length_cumsum_pmax_diff"]] = np.nan

    return df_y