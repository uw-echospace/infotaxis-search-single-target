from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pooch


DEFAULT_REPO = "uw-echospace/infotaxis-search-single-target"
DEFAULT_TAG = "v0.2.0a1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(
                    f"Unsafe path in archive {archive_path.name}: {member.name}"
                )
        tar.extractall(path=destination)


def _fetch_release_assets(repo: str, tag: str) -> list[dict]:
    api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    request = Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "infotaxis-data-prep-script",
        },
    )
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assets = payload.get("assets", [])
    return [asset for asset in assets if asset.get("name", "").endswith(".tar.gz")]


def _download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    if overwrite and destination.exists():
        destination.unlink()
    pooch.retrieve(
        url=url,
        known_hash=None,
        path=destination.parent,
        fname=destination.name,
        downloader=pooch.HTTPDownloader(headers={"User-Agent": "infotaxis-data-prep-script"}),
    )


def download_and_extract_release_assets(
    repo: str = DEFAULT_REPO,
    tag: str = DEFAULT_TAG,
    output_dir: Path | None = None,
    keep_archives: bool = False,
    overwrite: bool = False,
) -> None:
    if output_dir is None:
        output_dir = _repo_root() / "simulation_data"
    output_dir = output_dir.resolve()
    archives_dir = output_dir / "_archives"

    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir.mkdir(parents=True, exist_ok=True)

    try:
        assets = _fetch_release_assets(repo=repo, tag=tag)
    except HTTPError as err:
        raise RuntimeError(
            f"GitHub API request failed ({err.code}) for release tag {tag} in {repo}."
        ) from err
    except URLError as err:
        raise RuntimeError(f"Failed to reach GitHub API: {err.reason}") from err

    if not assets:
        raise RuntimeError(
            f"No .tar.gz assets found for release tag {tag} in repository {repo}."
        )

    for asset in assets:
        asset_name = asset["name"]
        asset_url = asset["browser_download_url"]
        archive_path = archives_dir / asset_name
        stamp_path = archives_dir / f".{asset_name}.extracted"

        should_extract = overwrite or not stamp_path.exists()
        should_download = should_extract and (overwrite or not archive_path.exists())

        if should_download:
            print(f"Downloading {asset_name}...")
            _download_file(url=asset_url, destination=archive_path, overwrite=overwrite)
        elif not should_extract:
            print(f"Skipping download (already extracted): {asset_name}")
        else:
            print(f"Skipping download (already exists): {asset_name}")

        if should_extract:
            print(f"Extracting {asset_name} into {output_dir}...")
            _safe_extract_tar(archive_path=archive_path, destination=output_dir)
            stamp_path.write_text("ok\n", encoding="utf-8")
        else:
            print(f"Skipping extract (already extracted): {asset_name}")

        if not keep_archives and archive_path.exists():
            archive_path.unlink()

    print(f"Done. Simulation data available in: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download all .tar.gz assets from a GitHub release tag and extract them into "
            "a single simulation_data folder."
        )
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repository in owner/repo format.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="GitHub release tag to download from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "simulation_data",
        help="Destination folder for extracted simulation data.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar.gz files in output_dir/_archives.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and re-extract assets even if local files exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_and_extract_release_assets(
        repo=args.repo,
        tag=args.tag,
        output_dir=args.output_dir,
        keep_archives=args.keep_archives,
        overwrite=args.overwrite,
    )