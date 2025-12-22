# Simulation and data preparation steps before plotting

## Randomized target location (Fig. 4)
- python script: `20251027_one_target_search_random_target.py`
- run parameter combinations created by `20251019_create_param_file_equal_pm_pfa.py` (i.e., use the same set of parameter files as runs saved in `20251026_revised_run`)
- raw output saved in folders:
    - `20251027`
    - `20251027_summary`
- to get data used in plotting notebook:
    - summary files (`_summary.csv` and `_ph.nc`) are generated together with raw output, so didn't need to run the summary extraction scripts
    - need to assemble summary from all runs for each parameter combinations via notebook `fig_4_data_assemble_summary_ph_csv.ipynb`
    - the summary files are futher refined by running `fig_4_data_infotaxis_MAP_refine_summary.ipynb`
        - output: `20251027_summary_refined`

### Infotaxis with mismatched PM and PFA between agent assumption and truth (Fig. 5)
- python script: `20251119_one_target_search_random_target_mismatch_agent_truth.py`
- run parameter combinations created by `20251120_create_param_file_equal_pm_pfa_mistmach_agent_truth.py`
- raw output saved in folders:
    - `20251119`
    - `20251119_summary`
- to get data used in plotting notebook:
    - summary files (`_summary.csv` and `_ph.nc`) are generated together with raw output, so didn't need to run the summary extraction scripts
    - need to assemble summary from all runs for each parameter combinations via notebook `fig_5_data_assemble_summary_ph_csv.ipynb`
    - the summary files are futher refined by running `fig_5_data_mismatch_agent_truth_refine_summary.ipynb`
        - output: `20251119_summary_refined`

## PM=0 with increasing search space size (Fig. 6B)
- python script: `20251027_one_target_search_random_target.py`
- run parameter combinations created by `20251028_create_param_file_change_cr.py`
- raw output saved in folder `20251028`
- to get data used in plotting notebook:
    - summary files (`_summary.csv` and `_ph.nc`) are generated together with raw output, so didn't need to run the summary extraction scripts
    - need to assemble summary from all runs for each parameter combinations via notebook `fig_6B_data_assemble_summary_ph_csv.ipynb`

## Infotaxis with restricted beam aim movement (Fig. S1)
- python script: `20251118_one_target_search_random_target_restrict_beam_movement.py`
- run parameter combinations created by `20251118_create_param_file_equal_pm_pfa.py`
- raw output saved in folders:
    - `20251118`
    - `20251118_summary`
- to get data used in plotting notebook:
    - summary files (`_summary.csv` and `_ph.nc`) are generated together with raw output, so didn't need to run the summary extraction scripts
    - need to assemble summary from all runs for each parameter combinations via notebook `fig_S1_data_assemble_summary_ph_csv.ipynb`
    - the summary files are futher refined by running `fig_S1_data_restrict_beam_movement_refine_summary.ipynb`
        - output: `20251118_summary_refined`

## PFA=0 with varying PM / PM=0 with varying PFA (Fig. S4 and S5)
- python script: `20251027_one_target_search_random_target.py`
- run parameter combinations created by:
    - `20251113_create_param_file_pfa0.py`
    - `20251113_create_param_file_pm0.py`
- raw output saved in folders:
    - `20251113`
    - `20251113_summary`
- to get data used in plotting notebook:
    - summary files (`_summary.csv` and `_ph.nc`) are generated together with raw output, so didn't need to run the summary extraction scripts
    - need to assemble summary from all runs for each parameter combinations via notebook `fig_S4_S5_data_assemble_summary_ph_csv.ipynb`
    - the summary files are futher refined by running `fig_S4_S5_infotaxis_MAP_pfa0_pm0_refine_summary.ipynb`
        - output: `20251113_summary_refined`

## Fixed target location
- python script: `20251025_one_target_search.py`
- run parameter combinations created by `20251019_create_param_file_equal_pm_pfa.py`
- raw output saved in folder `20251026_revised_run`
- summary files (`_summary.csv` and `_ph.nc`) saved in `20251026_revised_run_summary`, by running the following scripts to extract summary files from raw output: 
    - `20251026_run_stat_extract_infotaxis.py`
    - `20251026_run_stat_extract_MAP.py`
