"""Data download utilities.

This module provides functionality for downloading raw data from
various public sources for London, NYC, and other cities.
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import requests

from congestion_pricing.utils.io import ensure_dir, save_checksums
from congestion_pricing.utils.logging import get_logger

logger = get_logger(__name__)


# Data source URLs
DATA_SOURCES = {
    "london": {
        "osm": "https://download.geofabrik.de/europe/great-britain/england/greater-london-latest.osm.pbf",
        "gtfs": "https://tfl.gov.uk/tfl/syndication/feeds/gtfs.zip",
        # Note: Some TfL data requires API registration
    },
    "nyc": {
        "osm": "https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf",
        # Note: MTA GTFS requires acceptance of terms
        "taxi_zones": "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON",
    },
}


class DataDownloader:
    """Download raw data for a city."""

    def __init__(
        self,
        city: str,
        output_dir: str | Path = "data/raw",
        timeout: int = 300,
        retries: int = 3,
    ):
        """Initialize downloader.

        Args:
            city: City identifier ('london', 'nyc')
            output_dir: Root directory for raw data
            timeout: Request timeout in seconds
            retries: Number of retry attempts
        """
        self.city = city.lower()
        self.output_dir = Path(output_dir) / self.city
        self.timeout = timeout
        self.retries = retries
        self.checksums: dict[str, str] = {}

        if self.city not in DATA_SOURCES:
            raise ValueError(f"Unknown city: {city}. Available: {list(DATA_SOURCES.keys())}")

    def download_file(
        self,
        url: str,
        filename: str,
        force: bool = False,
    ) -> Path:
        """Download a single file.

        Args:
            url: URL to download
            filename: Output filename
            force: Whether to overwrite existing files

        Returns:
            Path to downloaded file
        """
        output_path = ensure_dir(self.output_dir) / filename

        if output_path.exists() and not force:
            logger.info(f"File exists, skipping: {filename}")
            return output_path

        logger.info(f"Downloading {filename} from {url}")

        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()

                # Stream to file
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                hash_obj = hashlib.sha256()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        hash_obj.update(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            pct = 100 * downloaded / total_size
                            if downloaded % (10 * 1024 * 1024) < 8192:  # Log every ~10MB
                                logger.info(f"  {pct:.0f}% ({downloaded / 1e6:.1f} MB)")

                self.checksums[filename] = hash_obj.hexdigest()
                logger.info(f"Downloaded {filename} ({downloaded / 1e6:.1f} MB)")
                return output_path

            except requests.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed to download {url}")

    def download_osm(self, force: bool = False) -> Path:
        """Download OpenStreetMap data.

        Args:
            force: Whether to overwrite existing files

        Returns:
            Path to downloaded file
        """
        url = DATA_SOURCES[self.city].get("osm")
        if not url:
            raise ValueError(f"No OSM URL configured for {self.city}")

        return self.download_file(url, f"{self.city}.osm.pbf", force)

    def download_gtfs(self, force: bool = False) -> Path | None:
        """Download GTFS transit data.

        Args:
            force: Whether to overwrite existing files

        Returns:
            Path to downloaded file, or None if not available
        """
        url = DATA_SOURCES[self.city].get("gtfs")
        if not url:
            logger.warning(f"No GTFS URL configured for {self.city}")
            return None

        return self.download_file(url, "gtfs.zip", force)

    def download_all(
        self,
        datasets: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Path]:
        """Download all configured datasets.

        Args:
            datasets: List of dataset names to download, or None for all
            force: Whether to overwrite existing files

        Returns:
            Dictionary mapping dataset name to path
        """
        available = DATA_SOURCES.get(self.city, {})
        if datasets is None:
            datasets = list(available.keys())

        results = {}
        for name in datasets:
            if name not in available:
                logger.warning(f"Dataset not available for {self.city}: {name}")
                continue

            url = available[name]
            ext = url.split(".")[-1].split("?")[0]
            filename = f"{name}.{ext}"

            try:
                results[name] = self.download_file(url, filename, force)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")

        # Save checksums
        if self.checksums:
            save_checksums(self.checksums, self.output_dir / "checksums.sha256")

        return results

    def create_manifest(self) -> dict[str, Any]:
        """Create a manifest of downloaded files.

        Returns:
            Manifest dictionary
        """
        import datetime

        manifest = {
            "city": self.city,
            "download_date": datetime.datetime.now().isoformat(),
            "sources": DATA_SOURCES.get(self.city, {}),
            "files": {},
        }

        for file_path in self.output_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                manifest["files"][file_path.name] = {
                    "size_bytes": file_path.stat().st_size,
                    "checksum": self.checksums.get(file_path.name, ""),
                }

        # Save manifest
        import yaml

        with open(self.output_dir / "manifest.yaml", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False)

        return manifest


def download_city_data(
    city: str,
    output_dir: str | Path = "data/raw",
    datasets: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Convenience function to download data for a city.

    Args:
        city: City identifier
        output_dir: Output directory
        datasets: Datasets to download
        force: Whether to overwrite existing

    Returns:
        Dictionary of downloaded files
    """
    downloader = DataDownloader(city, output_dir)
    results = downloader.download_all(datasets, force)
    downloader.create_manifest()
    return results
