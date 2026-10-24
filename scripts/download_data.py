"""Download public TDC source files, atomically, with a verifiable manifest."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

SOURCES = {
    "solubility": 4259610,
    "caco2": 4259569,
    "cyp2d6": 4259580,
}

EXPECTED_SHA256 = {
    "solubility": "0b593f7b55abbe477ba4284aefd7b5450551f4ed21ac4aed80a097931141c9a4",
    "caco2": "447d9f1af487c06b080145a2361dbe45af5b27afc83e1a975dd29b9ea535d6ab",
    "cyp2d6": "1b50a2f02a6d9c02d29fd9e3684d661f64bfe8afb1fea09e5ffb3d54174c411b",
}


def verify(name, raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256[name]:
        raise ValueError(f"Source checksum mismatch for {name}; refusing changed data")


def download(destination=Path("data/raw")):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, file_id in SOURCES.items():
        url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        path = destination / f"{name}.tab"
        if not path.exists():
            response = requests.get(url, timeout=180, headers={"User-Agent": "ADMET-portfolio-research/0.1"})
            response.raise_for_status()
            verify(name, response.content)
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(response.content)
            try:
                frame = pd.read_csv(temporary, sep="\t")
                assert {"Drug", "Y"} <= set(frame.columns), frame.columns
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        frame = pd.read_csv(path, sep="\t")
        assert {"Drug", "Y"} <= set(frame.columns)
        raw = path.read_bytes()
        verify(name, raw)
        manifest[name] = {"url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                          "rows": len(frame)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    download()
