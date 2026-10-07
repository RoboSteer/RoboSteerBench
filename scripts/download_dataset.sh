#!/usr/bin/env bash
# Download, verify, and extract the RoboSteer release one archive at a time.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/SteerableMotionBenchmarkDataset}"
mkdir -p -- "$DATASET_ROOT"
DATASET_ROOT="$(cd -- "$DATASET_ROOT" && pwd -P)"

STAGING="$DATASET_ROOT/.robosteer-download"
COMPLETED="$DATASET_ROOT/.robosteer-extracted-sha256"
if [[ -e "$STAGING" && ! -f "$STAGING/.robosteer-staging" ]]; then
  echo "Refusing to use an unrecognized staging directory: $STAGING" >&2
  exit 1
fi
mkdir -p -- "$STAGING"
touch -- "$STAGING/.robosteer-staging" "$COMPLETED"

REPO="YanCORANV/RoboSteer"
hf download "$REPO" SHA256SUMS --repo-type dataset --local-dir "$STAGING" >/dev/null
MANIFEST="$STAGING/SHA256SUMS"

count=0
while read -r checksum relative; do
  [[ -n "${checksum:-}" ]] || continue
  if [[ ! "$checksum" =~ ^[0-9a-f]{64}$ || ! "$relative" =~ ^archives/(Data/(Level[123]|Shared)|Tasks/Level[123]|root)/part-[0-9]{6}\.tar$ ]]; then
    echo "Invalid archive manifest entry: $checksum $relative" >&2
    exit 1
  fi
  ((count += 1))
  archive="$STAGING/$relative"
  marker="$checksum  $relative"
  if grep -Fqx -- "$marker" "$COMPLETED"; then
    rm -f -- "$archive"
    echo "[$count] Already extracted: $relative"
    continue
  fi
  echo "[$count] Downloading: $relative"
  hf download "$REPO" "$relative" --repo-type dataset --local-dir "$STAGING" >/dev/null
  if ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c - >/dev/null; then
    rm -f -- "$archive"
    echo "Checksum mismatch; rerun the script to download $relative again" >&2
    exit 1
  fi
  tar -xf "$archive" -C "$DATASET_ROOT"
  printf '%s\n' "$marker" >> "$COMPLETED"
  rm -f -- "$archive"
done < "$MANIFEST"

if ((count != 23)); then
  echo "Expected 23 release archives; found $count" >&2
  exit 1
fi
for required in Data/Shared/Motion Tasks/Level1 Tasks/Level2 Tasks/Level3; do
  if [[ ! -d "$DATASET_ROOT/$required" ]]; then
    echo "Extracted dataset is missing $required" >&2
    exit 1
  fi
done

rm -rf -- "$STAGING"
echo "Dataset ready: $DATASET_ROOT"
