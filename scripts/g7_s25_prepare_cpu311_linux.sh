#!/usr/bin/env bash
set -Eeuo pipefail
mode="$1";source_root="$2"
[[ "$mode" == preflight || "$mode" == install ]]||exit 2
sha="108808f7b685107a76fae1ce5605b7ff9e2bac27"
base="$HOME/.local/share/ttt-hybrid"
scope="$base/g7-s25-cpu311-r1"
r3="$base/g7-r3-06a01c2c3a98/venv/bin/python"
status="BLOCKED_LINUX_CPU311";full="NOT_RUN"
receipt="$source_root/local_runs/cpu311_install_linux.json"
if [[ "$mode" == install ]];then
 mkdir -p "$source_root/local_runs"
 finish(){
  rc=$?
  printf '{"schema":"ttt-hybrid.cpu311.scoped-linux.v1","source_sha":"%s","status":"%s","full_regression":"%s","r3":"UNCHANGED","cuda":"NOT_USED","scientific_first_use":"NOT_RUN"}\n' "$sha" "$status" "$full" >"$receipt"||:
  exit "$rc"
 }
 trap finish EXIT
fi
[[ "$(id -u)" -ne 0 ]]||{ echo LINUX_ROOT_REFUSED;exit 3; }
[[ "$(git -C "$source_root" rev-parse HEAD)" == "$sha" ]]||{ echo LINUX_SHA_DRIFT;exit 3; }
[[ -z "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]||{ echo LINUX_DIRTY_TRACKED_SOURCE;exit 3; }
[[ "$(git -C "$source_root" hash-object -- experiments/g7_s25_frozen_contradiction_ttt.json)" == "31633b341403729b7ffbb7d73f3a04098ad44380" ]]||{ echo LINUX_PROTOCOL_DRIFT;exit 3; }
[[ -x "$r3" && ! -e "$scope" ]]||{ echo LINUX_R3_OR_SCOPE_BLOCKED;exit 3; }
free="$(df -Pk "$HOME"|awk 'NR==2 {print $4}')"
[[ "$free" -ge 6291456 ]]||{ echo LINUX_DISK_RESERVE;exit 3; }
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0
manifest(){
 "$r3" -I -B - <<'PY'
import hashlib,json,importlib.metadata as m
p=sorted((str(d.metadata.get("Name","")).lower().replace("_","-"),str(d.version)) for d in m.distributions())
print(hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest())
PY
}
before="$(manifest)"
[[ "$before" == "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df" ]]||{ echo LINUX_R3_MANIFEST_DRIFT;exit 3; }
if [[ "$mode" == preflight ]];then echo LINUX_CPU311_PREFLIGHT=PASS;exit 0;fi
mkdir "$scope";chmod 700 "$scope"
export UV_PYTHON_INSTALL_DIR="$scope/pythons" UV_CACHE_DIR="$scope/cache" UV_NO_CONFIG=1
export UV_PYTHON_PREFERENCE=only-managed UV_PYTHON_INSTALL_BIN=false
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=1 UV_HTTP_RETRIES=1
archive="$scope/uv-v0.9.13.tar.gz"
curl --fail --location --silent --show-error --retry 0 --connect-timeout 15 --max-time 180 \
 "https://github.com/astral-sh/uv/releases/download/0.9.13/uv-x86_64-unknown-linux-gnu.tar.gz" --output "$archive"
printf '%s  %s\n' c45a44144bf23a2182e143227b4ad0bbe41a2bb7161a637c02e968906af53fd1 "$archive"|sha256sum -c -
mkdir "$scope/uv_release";tar -xzf "$archive" -C "$scope/uv_release"
uv="$(find "$scope/uv_release" -name uv -type f -print -quit)"
[[ -n "$uv" ]]||exit 3
chmod 700 "$uv"
uv_version="$("$uv" --version)"
[[ "$uv_version" == "uv 0.9.13" || "$uv_version" == "uv 0.9.13 "* ]]||exit 3
"$uv" python install 3.11.14
"$uv" venv --python 3.11.14 "$scope/venv"
python="$scope/venv/bin/python"
# Do not mix indexes: PyPI normal dependencies first, official CPU torch only.
"$uv" pip install --python "$python" --only-binary ':all:' \
 --index-url https://pypi.org/simple \
 'pytest==8.3.4' 'numpy==1.26.4' 'filelock>=3.13.1' 'typing-extensions>=4.8.0' \
 'networkx' 'jinja2' 'fsspec' 'sympy==1.13.1'
"$uv" pip install --python "$python" --only-binary ':all:' --no-deps \
 --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu'
"$uv" pip check --python "$python"
"$python" -I -B - <<'PY'
import sys,torch,pytest
assert sys.version_info[:3]==(3,11,14)
assert torch.__version__=="2.5.1+cpu"
assert pytest.__version__=="8.3.4"
assert torch.version.cuda is None
print("LINUX_CPU311_VERSION=PASS")
PY
export EXPECTED_CANDIDATE_SHA="$sha"
"$python" -I -B "$source_root/scripts/qualify_cpu.py" --receipt "$source_root/local_runs/s25_full_regression_linux.json"
"$python" -I -B - "$source_root/local_runs/s25_full_regression_linux.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["status"]=="PASS_CPU_ENGINEERING_ONLY"
assert r["revision"]=="108808f7b685107a76fae1ce5605b7ff9e2bac27"
assert r["negative_control"]["pass"] is True
print("LINUX_CPU311_REGRESSION=PASS")
PY
[[ "$(manifest)" == "$before" ]]||{ echo LINUX_R3_CHANGED;exit 3; }
status="PASS_LINUX_CPU311";full="PASS_EXACT_S25_FULL_SUITE_AND_NEGATIVE_CONTROL"
echo "LINUX_CPU311=$status"
