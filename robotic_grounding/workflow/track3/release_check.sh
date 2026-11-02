#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

status=0
fail() {
  echo "ERROR: $*" >&2
  status=1
}

for required in README.md LICENSE THIRD_PARTY_NOTICES.md reconstruction robotic_grounding; do
  [[ -e "${required}" ]] || fail "missing ${required}"
done

python_files=()
while IFS= read -r -d '' path; do python_files+=("${path}"); done < <(find reconstruction robotic_grounding -type f -name '*.py' -print0)
python - "${python_files[@]}" <<'PY' || status=1
import ast
import pathlib
import sys

errors = []
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        errors.append(f"{path}: {exc}")
if errors:
    print("Python parse failures:", file=sys.stderr)
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Python parse PASS ({len(sys.argv) - 1} files)")
PY

while IFS= read -r -d '' script; do
  bash -n "${script}" || status=1
done < <(find reconstruction robotic_grounding -type f \( -name '*.sh' -o -name '*.sbatch' \) -print0)

credential_files=$(find . -type f \( -name 'kaggle.json' -o -name '.env' -o -name '*.pem' -o -name 'id_rsa' -o -name 'id_ed25519' \) -print)
[[ -z "${credential_files}" ]] || fail "credential-like files found: ${credential_files}"

private_paths=$(rg -l '/scratch/project/|/scratch/user/[^$]|/home/[^/$]+/|/root/workspace/' . \
  --glob '!robotic_grounding/workflow/track3/release_check.sh' \
  --glob '!robotic_grounding/third_party/v2d_submission_kit/**' || true)
[[ -z "${private_paths}" ]] || fail "non-portable absolute paths found: ${private_paths}"

forbidden_assets=$(find . -type f \( -name '*.safetensors' -o -name '*.onnx' -o -name '*.urdf' \
  -o -name '*.glb' -o -name '*.stl' -o -name 'MANO_LEFT.pkl' -o -name 'MANO_RIGHT.pkl' \
  -o -name 'mano_mean_params.npz' -o -name 'V_regressor_84.npy' -o -name 'V_regressor_336.npy' \) -print)
[[ -z "${forbidden_assets}" ]] || fail "generated/licensed assets found: ${forbidden_assets}"

large_files=$(find . -type f -size +20M -print)
[[ -z "${large_files}" ]] || fail "files larger than 20 MiB found: ${large_files}"

if [[ ${status} -ne 0 ]]; then
  echo "Release check FAILED" >&2
  exit ${status}
fi
echo "Shell syntax PASS"
echo "Credential/path/asset/size checks PASS"
echo "Release check PASS (no end-to-end execution performed)"

