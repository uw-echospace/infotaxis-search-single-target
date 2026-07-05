# Infotaxis for echolocation-based target search

This repository contains code associated with the following paper:

**Modeling echolocation as an active pursuit of information via infotaxis**

Authors: [Wu-Jung Lee](https://uw-echospace.github.io/author/wu-jung-lee/) ([@leewujung](https://github.com/leewujung)), John R. Buck, and Peter L. Tyack


## Repo structure
- [`src/infotaxis`](./src/infotaxis/): Core model code for an echolocating infotaxis agent searching for a single target
- [`src/data_prep_plot`](./src/data_prep_plot/): Helper modules for data preparation and plotting
- [`docs`](./docs/): Inventory for simulation data preparation scripts and fig notebooks
- [`notebooks_fig`](./notebooks_fig/): Notebooks to generate figures in the paper
- [`scripts_data_scripts`](./scripts_data_prep/): Scripts to prepare simulation data for plotting
- [`scripts_simulation`](./scripts_simulation): Scripts to run infotaxis simulation


## Download simulation data from GitHub release assets

To download all `.tar.gz` raw simulation archives from release tag `v0.2.0a1` and extract
them into a single root-level folder named `simulation_data`, run:

```bash
python src/data_prep_plot/download_simulation_data.py
```

Default behavior:
- Reads release assets from `uw-echospace/infotaxis-search-single-target` tag `v0.2.0a1`
- Downloads only `.tar.gz` assets
- Extracts all archives into `./simulation_data`
- Removes downloaded archives after extraction

Useful options:

```bash
# Keep downloaded archives in simulation_data/_archives
python src/data_prep_plot/download_simulation_data.py --keep-archives

# Redownload and re-extract everything
python src/data_prep_plot/download_simulation_data.py --overwrite

# Use a different release tag or repository
python src/data_prep_plot/download_simulation_data.py --tag <tag> --repo <owner/repo>
```
