#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 input_path output_directory" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_root/.conda/gandalf/bin/python"

input_path="$1"
output_directory="$2"

"$python_bin" "$repo_root/scripts/build_graph.py" \
  --backend both \
  --input "$input_path" \
  --output "$output_directory"
