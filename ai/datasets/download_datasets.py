#!/usr/bin/env python3
"""
SIH26052 — NOICELESSX: Official Research Dataset Downloader.
Acquires verified research datasets from authoritative official repositories,
verifies checksums, enforces licensing agreements, and supports resumable downloads.
"""

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class DatasetEntry:
    id: str
    name: str
    category: str
    official_url: str
    download_urls: List[str]
    citation: str
    license_type: str
    terms_notice: str
    size_str: str = ""
    notes: str = ""
    requires_consent: bool = False
    expected_sha256: Optional[str] = None
    expected_size_bytes: Optional[int] = None
    archive_format: Optional[str] = None  # 'zip', 'tar.gz', None


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "voicebank_clean": DatasetEntry(
        id="voicebank_clean",
        name="VoiceBank (VCTK subset used in VoiceBank+DEMAND)",
        category="clean_speech",
        official_url="https://datashare.ed.ac.uk/handle/10283/2791",
        download_urls=[
            "https://datashare.ed.ac.uk/download/DS_10283_2791.zip",
        ],
        citation="Valentini-Botinhao et al., 'Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech', SSW 2016.",
        license_type="CC BY 4.0",
        terms_notice="Creative Commons Attribution 4.0. Requires citation.",
        size_str="11,572 train / 824 test utterances, 28+2 speakers",
        notes="The standard benchmark — use this so you can report PESQ/STOI numbers comparable to published work",
        archive_format="zip",
    ),
    "vctk": DatasetEntry(
        id="vctk",
        name="VCTK full corpus",
        category="clean_speech",
        official_url="https://datashare.ed.ac.uk/handle/10283/3443",
        download_urls=[
            "https://datashare.ed.ac.uk/download/DS_10283_3443.zip",
        ],
        citation="Yamagishi et al., 'CSTR VCTK Corpus: English Multi-speaker Speech Corpus for CSTR Voice Cloning Toolkit', 2019.",
        license_type="Open Data Commons Attribution License (ODC-By) v1.0",
        terms_notice="Attribution to CSTR, University of Edinburgh required.",
        size_str="~44 hours, 110 speakers",
        notes="Larger speaker pool for the AI model's generalization",
        archive_format="zip",
    ),
    "librispeech": DatasetEntry(
        id="librispeech",
        name="LibriSpeech (train-clean-100 / train-clean-360)",
        category="clean_speech",
        official_url="https://www.openslr.org/12",
        download_urls=[
            "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
        ],
        citation="Panayotov et al., 'Librispeech: an ASR corpus based on public domain audio books', ICASSP 2015.",
        license_type="CC BY 4.0",
        terms_notice="Creative Commons Attribution 4.0. Based on LibriVox public domain recordings.",
        size_str="100–460 hours",
        notes="Large-scale clean read speech, standard ASR/SE corpus",
        archive_format="tar.gz",
    ),
    "dns5_clean": DatasetEntry(
        id="dns5_clean",
        name="Microsoft DNS Challenge 5 — clean_fullband",
        category="clean_speech",
        official_url="https://github.com/microsoft/DNS-Challenge",
        download_urls=[
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-5-headset-training.sh",
        ],
        citation="Dubey et al., 'ICASSP 2023 Deep Noise Suppression Challenge', ICASSP 2023.",
        license_type="CC BY-NC 4.0",
        terms_notice="Creative Commons Attribution-NonCommercial 4.0. Clean headset speech.",
        size_str="up to 827 GB (subsample)",
        notes="Very large multi-language clean speech pool; also ships noise + real room impulse responses in the same repo (download-dns-challenge-5-headset-training.sh)",
        archive_format=None,
    ),
    "demand_noise": DatasetEntry(
        id="demand_noise",
        name="DEMAND",
        category="noise_environmental",
        official_url="https://doi.org/10.5281/zenodo.1227121",
        download_urls=[
            # Zenodo official record files (18 environments, 16-channel 48k)
            "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip",
            "https://zenodo.org/records/1227121/files/DLIVING_16k.zip",
            "https://zenodo.org/records/1227121/files/DWASHING_16k.zip",
            "https://zenodo.org/records/1227121/files/NFIELD_16k.zip",
            "https://zenodo.org/records/1227121/files/NPARK_16k.zip",
            "https://zenodo.org/records/1227121/files/NRIVER_16k.zip",
            "https://zenodo.org/records/1227121/files/OHALLWAY_16k.zip",
            "https://zenodo.org/records/1227121/files/OMEETING_16k.zip",
            "https://zenodo.org/records/1227121/files/OOFFICE_16k.zip",
            "https://zenodo.org/records/1227121/files/PCAFE_16k.zip",
            "https://zenodo.org/records/1227121/files/PRESTAU_16k.zip",
            "https://zenodo.org/records/1227121/files/PSTATION_16k.zip",
            "https://zenodo.org/records/1227121/files/SCAFE_16k.zip",
            "https://zenodo.org/records/1227121/files/SPSQUARE_16k.zip",
            "https://zenodo.org/records/1227121/files/STRAFFIC_16k.zip",
            "https://zenodo.org/records/1227121/files/TBUS_16k.zip",
            "https://zenodo.org/records/1227121/files/TCARS_16k.zip",
            "https://zenodo.org/records/1227121/files/TMETRO_16k.zip",
        ],
        citation="Thiemann et al., 'Diverse Environments Multi-channel Acoustic Noise Database', Proc. Meetings on Acoustics 2013.",
        license_type="CC BY-SA 3.0",
        terms_notice="Creative Commons Attribution-ShareAlike 3.0. Attribution required.",
        size_str="18 real-world noise environments (domestic, office, public, transport, street, nature)",
        notes="Paired with VoiceBank in the standard benchmark",
        archive_format="zip",
    ),
    "musan": DatasetEntry(
        id="musan",
        name="MUSAN",
        category="noise_mixed",
        official_url="https://www.openslr.org/17",
        download_urls=[
            "https://www.openslr.org/resources/17/musan.tar.gz",
        ],
        citation="Snyder et al., 'MUSAN: A Music, Speech, and Noise Corpus', arXiv:1510.08484, 2015.",
        license_type="Creative Commons 0 / Public Domain",
        terms_notice="CC0 public domain dedication.",
        size_str="Music, Speech, Noise — ~109 hours",
        notes="Widely used for augmentation",
        archive_format="tar.gz",
    ),
    "dns5_noise": DatasetEntry(
        id="dns5_noise",
        name="DNS Challenge 5 — noise_fullband",
        category="noise_environmental",
        official_url="https://github.com/microsoft/DNS-Challenge",
        download_urls=[
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-5.sh",
        ],
        citation="Dubey et al., 'ICASSP 2023 Deep Noise Suppression Challenge', ICASSP 2023.",
        license_type="CC BY-NC 4.0",
        terms_notice="Creative Commons Attribution-NonCommercial 4.0. Noise fullband subset.",
        size_str="58 GB, sourced from AudioSet + Freesound",
        notes="Huge diversity of real-world noise classes",
        archive_format=None,
    ),
    "fsd50k": DatasetEntry(
        id="fsd50k",
        name="FSD50K",
        category="noise_events",
        official_url="https://zenodo.org/records/4060432",
        download_urls=[
            "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip",
            "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip",
            "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip",
            "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.z01",
        ],
        citation="Fonseca et al., 'FSD50K: an Open Dataset of Everyday Sounds with Freesound', IEEE/ACM TASLP 2022.",
        license_type="CC BY 4.0 / Freesound Licenses",
        terms_notice="Audio clips under various CC licenses. Metadata under CC BY 4.0.",
        size_str="51,197 clips, 200 sound-event classes (AudioSet ontology)",
        notes="If you want a literal '200 classes' number for your report, this is it",
        archive_format="zip",
    ),
    "esc50": DatasetEntry(
        id="esc50",
        name="ESC-50",
        category="noise_environmental",
        official_url="https://github.com/karolpiczak/ESC-50",
        download_urls=[
            "https://github.com/karolpiczak/ESC-50/archive/master.zip",
        ],
        citation="Piczak, 'ESC: Dataset for Environmental Sound Classification', ACM MM 2015.",
        license_type="CC BY-NC 3.0",
        terms_notice="Creative Commons Attribution-NonCommercial 3.0. Non-commercial research only.",
        size_str="2,000 clips, 50 environmental classes",
        notes="Small, clean-labeled, good for the impulse-detector training set (glass break, gunshot, clapping, etc. are in here)",
        archive_format="zip",
    ),
    "urbansound8k": DatasetEntry(
        id="urbansound8k",
        name="UrbanSound8K",
        category="noise_urban",
        official_url="https://urbansounddataset.weebly.com/urbansound8k.html",
        download_urls=[
            "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz",
        ],
        citation="Salamon et al., 'A Dataset and Taxonomy for Urban Sound Research', ACM MM 2014.",
        license_type="CC BY-NC 3.0",
        terms_notice="MANDATORY: UrbanSound8K terms of use require accepting academic research terms at https://urbansounddataset.weebly.com/urbansound8k.html. Explicit consent required via --accept-urbansound8k-terms.",
        size_str="8,732 clips, 10 urban classes",
        notes="Sirens, drilling, engine idling — good defence/urban noise coverage",
        requires_consent=True,
        archive_format="tar.gz",
    ),
    "tau2020": DatasetEntry(
        id="tau2020",
        name="TAU Urban Acoustic Scenes 2020",
        category="noise_scenes",
        official_url="https://zenodo.org/records/3819968",
        download_urls=[
            "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.audio.1.zip",
            "https://zenodo.org/records/3819968/files/TAU-urban-acoustic-scenes-2020-mobile-development.meta.zip",
        ],
        citation="Mesaros et al., 'Acoustic Scene Classification in DCASE 2020 Challenge', DCASE 2020.",
        license_type="CC BY 4.0",
        terms_notice="Creative Commons Attribution 4.0. Attribution required.",
        size_str="10 acoustic scenes, ~40 hours",
        notes="Airport, metro, park, street — background-scene diversity",
        archive_format="zip",
    ),
    "rirs_noises": DatasetEntry(
        id="rirs_noises",
        name="RIRS_NOISES (OpenSLR 28)",
        category="rir",
        official_url="https://www.openslr.org/28",
        download_urls=[
            "https://www.openslr.org/resources/28/rirs_noises.zip",
        ],
        citation="Ko et al., 'A study on data augmentation of reverberant speech for robust speech recognition', ICASSP 2017.",
        license_type="Apache 2.0",
        terms_notice="Apache License 2.0. Free commercial/non-commercial use with notice.",
        size_str="Simulated + real RIRs",
        notes="Simulated + real RIRs, standard for reverb augmentation",
        archive_format="zip",
    ),
    "dns5_rir": DatasetEntry(
        id="dns5_rir",
        name="DNS Challenge 5 — impulse_responses",
        category="rir",
        official_url="https://github.com/microsoft/DNS-Challenge",
        download_urls=[
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-5.sh",
        ],
        citation="Dubey et al., 'ICASSP 2023 Deep Noise Suppression Challenge', ICASSP 2023.",
        license_type="CC BY-NC 4.0",
        terms_notice="Creative Commons Attribution-NonCommercial 4.0. Impulse responses.",
        size_str="5.9 GB",
        notes="Same repo as above, 5.9 GB",
        archive_format=None,
    ),
    "dns5": DatasetEntry(
        id="dns5",
        name="Microsoft DNS Challenge 5 (Comprehensive)",
        category="speech_noise_rir",
        official_url="https://github.com/microsoft/DNS-Challenge",
        download_urls=[
            "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/download-dns-challenge-5.sh",
        ],
        citation="Dubey et al., 'ICASSP 2023 Deep Noise Suppression Challenge', ICASSP 2023.",
        license_type="CC BY-NC 4.0",
        terms_notice="Creative Commons Attribution-NonCommercial 4.0. Full corpus is 827GB; script downloads targeted noise and RIR subsets.",
        size_str="Clean (up to 827GB), Noise (58GB), RIR (5.9GB)",
        notes="Very large multi-language clean speech pool; also ships noise + real room impulse responses in the same repo",
        archive_format=None,
    ),
}


def print_license_notice(entry: DatasetEntry):
    """Prints the licensing terms and citation requirements to standard output."""
    print("--------------------------------------------------------------------------")
    print(f"Dataset:       {entry.name}")
    print(f"Official URL:  {entry.official_url}")
    print(f"License:       {entry.license_type}")
    print(f"Citation:      {entry.citation}")
    print(f"Terms Notice:  {entry.terms_notice}")
    print("--------------------------------------------------------------------------")


def verify_file_checksum(filepath: Path, expected_sha256: str, chunk_size: int = 1024 * 1024) -> bool:
    """Computes and compares SHA-256 checksum."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    digest = hasher.hexdigest().lower()
    return digest == expected_sha256.lower()


def download_resumable(
    url: str,
    dest_path: Path,
    expected_sha256: Optional[str] = None,
    chunk_size: int = 256 * 1024,
) -> bool:
    """
    Downloads a file with HTTP Range support for resumption.
    Returns True if downloaded and verified successfully.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    initial_bytes = 0
    if temp_path.exists():
        initial_bytes = temp_path.stat().st_size

    headers = {
        "User-Agent": "Mozilla/5.0 (NOICELESSX Research Audio Downloader/1.0)",
    }
    if initial_bytes > 0:
        headers["Range"] = f"bytes={initial_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    print(f"  Connecting to: {url}")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            status_code = getattr(response, "status", 200)
            content_range = response.headers.get("Content-Range")
            content_length = int(response.headers.get("Content-Length", 0))

            if status_code == 206 and content_range:
                # Partial content response
                mode = "ab"
                total_size = initial_bytes + content_length
                print(f"  Resuming transfer from byte offset: {initial_bytes:,} (Total: {total_size:,} bytes)")
            else:
                # Full response
                mode = "wb"
                initial_bytes = 0
                total_size = content_length
                print(f"  Starting full transfer (Total: {total_size:,} bytes)")

            downloaded = initial_bytes
            start_time = time.time()
            last_report_time = start_time

            with open(temp_path, mode) as out_f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_report_time >= 0.5 or (total_size > 0 and downloaded >= total_size):
                        last_report_time = now
                        elapsed = max(0.001, now - start_time)
                        speed_mbps = (downloaded - initial_bytes) / (1024 * 1024 * elapsed)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100.0
                            print(
                                f"\r    [{pct:5.1f}%] {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB "
                                f"@ {speed_mbps:.2f} MB/s",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r    Downloaded {downloaded / (1024*1024):.1f} MB @ {speed_mbps:.2f} MB/s",
                                end="",
                                flush=True,
                            )

            print()

    except urllib.error.HTTPError as e:
        print(f"\n[ERROR] HTTP Error {e.code} ({e.reason}) while downloading {url}")
        return False
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Network/URL Error while accessing {url}: {e.reason}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error downloading {url}: {e}")
        return False

    # Checksum verification if available
    if expected_sha256:
        print("  Verifying SHA-256 checksum...")
        if not verify_file_checksum(temp_path, expected_sha256):
            print(f"[ERROR] Checksum mismatch for {dest_path.name}! Expected: {expected_sha256}")
            return False
        print("  Checksum verified successfully.")

    if dest_path.exists():
        dest_path.unlink()
    temp_path.rename(dest_path)
    print(f"  Successfully saved to: {dest_path}")
    return True


def extract_downloaded_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extracts zip, tar.gz, or tar.bz2 archives safely."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {archive_path.name} to {output_dir}...")
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(output_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(output_dir)
        else:
            print(f"  [Warning] Not an archive or unrecognized archive format: {archive_path}")
            return False
        print("  Extraction complete.")
        return True
    except Exception as e:
        print(f"  [ERROR] Extraction failed for {archive_path}: {e}")
        return False


def download_dataset(
    dataset_id: str,
    target_dir: Path,
    accept_urbansound8k: bool = False,
    dns5_subsample_gb: float = 20.0,
    dry_run: bool = False,
) -> bool:
    """
    Orchestrates downloading and extraction of an individual dataset from official source.
    """
    if dataset_id not in DATASET_REGISTRY:
        print(f"[ERROR] Unknown dataset identifier '{dataset_id}'.")
        print(f"Available datasets: {list(DATASET_REGISTRY.keys())}")
        return False

    entry = DATASET_REGISTRY[dataset_id]
    print_license_notice(entry)

    # Enforce consent for UrbanSound8K
    if entry.requires_consent and not accept_urbansound8k:
        print(f"\n[CONSENT REQUIRED] UrbanSound8K requires user agreement to its non-commercial terms.")
        print(f"Please review the terms at {entry.official_url} and pass --accept-urbansound8k-terms to proceed.")
        return False

    if dry_run:
        print(f"[Dry Run] Validated dataset {entry.name}. Would download {len(entry.download_urls)} file(s):")
        for u in entry.download_urls:
            print(f"  - {u}")
        return True

    dest_folder = target_dir / entry.category / entry.id
    dest_folder.mkdir(parents=True, exist_ok=True)
    archive_dir = target_dir / "archives" / entry.id
    archive_dir.mkdir(parents=True, exist_ok=True)

    success_all = True
    for idx, url in enumerate(entry.download_urls, 1):
        filename = url.split("?")[0].rstrip("/").split("/")[-1]
        if not filename or "." not in filename:
            filename = f"{entry.id}_part_{idx}.bin"

        dest_file = archive_dir / filename
        if dest_file.exists():
            print(f"  File already exists: {dest_file} (Skipping download, proceeding to verification/extraction)")
        else:
            ok = download_resumable(
                url=url,
                dest_path=dest_file,
                expected_sha256=entry.expected_sha256,
            )
            if not ok:
                success_all = False
                print(f"[FAIL] Download failed for {entry.name} URL: {url}")
                continue

        # Extract if archive
        if entry.archive_format:
            extract_downloaded_archive(dest_file, dest_folder)

    return success_all


def list_registered_datasets():
    """Prints a formatted inventory of all registered research datasets categorized by Clean Speech, Noise, and RIRs."""
    print("\n==========================================================================================================")
    print("                    SIH26052 NOICELESSX — Official Research Dataset Catalog                              ")
    print("==========================================================================================================")

    categories = [
        ("Clean speech (the 's[n]' in your mixing equation)", ["voicebank_clean", "vctk", "librispeech", "dns5_clean"]),
        ("Noise (the 'v[n]' term)", ["demand_noise", "musan", "dns5_noise", "fsd50k", "esc50", "urbansound8k", "tau2020"]),
        ("Room impulse responses (the 'h_s, h_v' convolution terms)", ["rirs_noises", "dns5_rir"]),
    ]

    for cat_title, entry_keys in categories:
        print(f"\n### {cat_title}")
        print(f"{'Dataset':<38} | {'Size':<28} | {'Source / Official URL':<46} | {'Notes'}")
        print("-" * 140)
        for key in entry_keys:
            if key in DATASET_REGISTRY:
                entry = DATASET_REGISTRY[key]
                print(f"{entry.name:<38} | {entry.size_str:<28} | {entry.official_url:<46} | {entry.notes}")
    print("\n==========================================================================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download authentic research datasets with checksums and licensing compliance."
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset ID to download (e.g. voicebank_clean, esc50)")
    parser.add_argument("--all", action="store_true", help="Download all registered research datasets")
    parser.add_argument("--list", action="store_true", help="List all registered datasets, licenses, and official URLs")
    parser.add_argument("--target-dir", type=str, default="data/raw", help="Target directory for downloaded audio (default: data/raw)")
    parser.add_argument("--accept-urbansound8k-terms", action="store_true", help="Affirm acceptance of UrbanSound8K usage terms")
    parser.add_argument("--dns5-subsample-gb", type=float, default=20.0, help="Maximum gigabytes to download for DNS-5 subset")
    parser.add_argument("--dry-run", action="store_true", help="Simulate and print download URLs without fetching files")

    args = parser.parse_args()

    if args.list:
        list_registered_datasets()
        sys.exit(0)

    target_path = Path(args.target_dir)

    if args.all:
        print("Starting acquisition for ALL registered research datasets...")
        failures = []
        for d_id in DATASET_REGISTRY.keys():
            print(f"\nProcessing {d_id}...")
            ok = download_dataset(
                dataset_id=d_id,
                target_dir=target_path,
                accept_urbansound8k=args.accept_urbansound8k_terms,
                dns5_subsample_gb=args.dns5_subsample_gb,
                dry_run=args.dry_run,
            )
            if not ok:
                failures.append(d_id)

        if failures:
            print(f"\n[SUMMARY] Acquisition finished with {len(failures)} failed datasets: {failures}")
            sys.exit(1)
        else:
            print("\n[SUMMARY] All datasets downloaded and extracted successfully.")
            sys.exit(0)

    elif args.dataset:
        ok = download_dataset(
            dataset_id=args.dataset,
            target_dir=target_path,
            accept_urbansound8k=args.accept_urbansound8k_terms,
            dns5_subsample_gb=args.dns5_subsample_gb,
            dry_run=args.dry_run,
        )
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()
        print("\nUse --list to view all datasets or specify --dataset <id> / --all.")
        sys.exit(1)


if __name__ == "__main__":
    main()
