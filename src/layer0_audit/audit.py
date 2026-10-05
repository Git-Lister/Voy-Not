"""Foundational audit module. Filled by Brick 1, updated in Brick 2."""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(payload: object) -> str:
    """Stable SHA-256 of a JSON-serializable payload. Uses sort_keys=True."""
    serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_hf_dataset(name: str, repo_id: str, config: str | None = None) -> dict:
    """Load a Hugging Face dataset. Returns status dict with keys:
    name, source_type, repo_id, config, status, row_count, schema, sha256, error."""
    from datasets import load_dataset

    status_dict: dict[str, object] = {
        "name": name,
        "source_type": "huggingface",
        "repo_id": repo_id,
        "config": config,
        "status": "failed",
        "row_count": 0,
        "schema": {},
        "sha256": None,
        "error": None,
    }

    try:
        config_str = f" (config={config})" if config else ""
        logger.info(f"Loading HF dataset: {repo_id}{config_str}")
        ds = load_dataset(repo_id, config, split="train")
        row_count = len(ds)
        columns = ds.column_names
        schema = {col: str(ds.features[col].dtype) if hasattr(ds.features[col], "dtype") else str(type(ds.features[col])) for col in columns}

        # Convert to list of dicts for hashing
        data_list = [row for row in ds]
        sha256 = compute_sha256(data_list)

        # Save raw data if small enough
        raw_dir = Path("data/raw") / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_file = raw_dir / "data.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)

        status_dict.update({
            "status": "loaded",
            "row_count": row_count,
            "schema": schema,
            "sha256": sha256,
        })
        logger.info(f"Successfully loaded {name}: {row_count} rows, schema: {schema}")

    except Exception as e:  # noqa: BLE001
        status_dict["error"] = str(e)
        logger.error(f"Failed to load {name}: {e}")

    return status_dict


def fetch_yale_coordinates() -> dict:
    """Attempt to fetch YaleDHLab coordinates from GitHub and Beinecke IIIF.
    Returns status dict with name, source_type, url, status, row_count, schema, sha256, error, discovered_alternates."""
    import urllib.request

    # Group A — GitHub (archived YaleDHLab repo and mirrors)
    github_urls = [
        "https://raw.githubusercontent.com/YaleDHLab/voynich/master/data/coordinates.json",
        "https://raw.githubusercontent.com/YaleDHLab/voynich/main/data/coordinates.json",
        "https://raw.githubusercontent.com/YaleDHLab/voynich/master/voynich_coordinates.json",
        "https://raw.githubusercontent.com/YaleDHLab/voynich/master/data/coordinates.csv",
        "https://raw.githubusercontent.com/YaleDHLab/voynich/master/voynich_data.json",
    ]

    # Group B — Beinecke IIIF (authoritative, live) - multiple URL attempts with User-Agent
    iiif_urls = [
        "https://collections.library.yale.edu/iiif/2/2002046/manifest",
        "https://collections.library.yale.edu/iiif/3/2002046/manifest",
        "https://collections.library.yale.edu/catalog/2002046.json",
        "https://collections.library.yale.edu/catalog/2002046/manifest",
        "https://collections.library.yale.edu/iiif/2/2002046",
    ]

    # Group C — Zenodo / archive mirrors (probe only)
    zenodo_url = "https://zenodo.org/api/records?q=voynich+coordinates&size=5"

    def fetch_with_ua(url: str, timeout: int = 30) -> dict | None:
        """Fetch URL with User-Agent header."""
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; VoynichApparatus/1.0)"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return None

    status_dict: dict[str, object] = {
        "name": "yale_coordinates",
        "source_type": "github",
        "url": None,
        "status": "unavailable",
        "row_count": 0,
        "schema": {},
        "sha256": None,
        "error": None,
        "discovered_alternates": [],
    }

    # Try Group A
    for url in github_urls:
        try:
            logger.info(f"Attempting to fetch Yale coordinates from: {url}")
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            status_dict["url"] = url
            status_dict["status"] = "loaded"
            status_dict["row_count"] = len(data) if isinstance(data, list) else 1
            if isinstance(data, list) and data:
                status_dict["schema"] = {k: type(v).__name__ for k, v in data[0].items()}
            else:
                status_dict["schema"] = {k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else {}
            status_dict["sha256"] = compute_sha256(data)
            logger.info(f"Successfully fetched Yale coordinates from {url}")
            return status_dict
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch from {url}: {e}")
            status_dict["error"] = str(e)

    # Try Group B — Beinecke IIIF (multiple URLs with User-Agent)
    for url in iiif_urls:
        try:
            logger.info(f"Attempting to fetch Beinecke IIIF manifest from: {url}")
            data = fetch_with_ua(url)
            if data is not None:
                # Save raw IIIF manifest
                raw_dir = Path("data/raw/beinecke_iiif")
                raw_dir.mkdir(parents=True, exist_ok=True)
                slug = url.split("/")[-2] if url.endswith("/manifest") else url.split("/")[-1]
                raw_file = raw_dir / f"{slug}.json"
                with open(raw_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                status_dict["name"] = "beinecke_iiif"
                status_dict["source_type"] = "iiif"
                status_dict["url"] = url
                status_dict["status"] = "partial"
                status_dict["row_count"] = 1
                status_dict["schema"] = {k: type(v).__name__ for k, v in data.items()} if isinstance(data, dict) else {}
                status_dict["sha256"] = compute_sha256(data)
                status_dict["error"] = None
                logger.info(f"Successfully fetched Beinecke IIIF manifest from {url}")
                return status_dict
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch from {url}: {e}")
            status_dict["error"] = str(e)

    # Try Group C — Zenodo (probe only)
    try:
        logger.info(f"Probing Zenodo for alternatives: {zenodo_url}")
        zenodo_data = fetch_with_ua(zenodo_url)
        if zenodo_data:
            hits = zenodo_data.get("hits", {}).get("hits", [])
            alternates = []
            for hit in hits[:5]:
                alternates.append({
                    "title": hit.get("metadata", {}).get("title", "Unknown"),
                    "doi": hit.get("doi", "Unknown"),
                    "url": hit.get("links", {}).get("self", "Unknown"),
                })
            status_dict["discovered_alternates"] = alternates
            logger.info(f"Found {len(alternates)} alternate records on Zenodo")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to probe Zenodo: {e}")
        status_dict["error"] = str(e)

    return status_dict


def generate_manifest(sources: list[dict]) -> dict:
    """Assemble the manifest dict. Writes to data/processed/data_manifest.json
    and returns the dict."""
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "cross_transcription_stats": {
            "source": "vcat_mismatch",
            "exact_match_pct": None,
            "normalized_match_pct": None,
            "high_similarity_pct": None,
            "substantive_disagreement_pct": None,
            "note": "Populate if the mismatch dataset exposes these fields; otherwise leave null.",
        },
    }

    # Try to populate cross_transcription_stats from mismatch data
    for src in sources:
        if src.get("name") == "vcat_mismatch" and src.get("status") == "loaded":
            # The mismatch dataset may have these fields - we'd need to compute them
            # For now, leave as null as specified
            pass

    output_path = Path("data/processed/data_manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(f"Manifest written to {output_path}")
    return manifest


def run_audit() -> dict:
    """Orchestrate the full audit. Returns the manifest dict."""
    logger.info("Starting foundational audit")

    sources = []

    # Load Hugging Face datasets
    # EVA and mismatch don't need config
    sources.append(load_hf_dataset("vcat_eva", "Ched-ai/voynich-eva"))
    sources.append(load_hf_dataset("vcat_mismatch", "Ched-ai/voynich-transcription-mismatch"))

    # Metadata has 3 configs - load each
    for config in ["pages", "folios", "quires"]:
        sources.append(load_hf_dataset(f"vcat_metadata_{config}", "Ched-ai/voynich-manuscript-metadata", config=config))

    # Fetch Yale coordinates with fallback chain
    sources.append(fetch_yale_coordinates())

    # Generate manifest
    manifest = generate_manifest(sources)

    logger.info("Audit complete")
    return manifest


if __name__ == "__main__":
    import json
    manifest = run_audit()
    print(json.dumps(manifest, indent=2))