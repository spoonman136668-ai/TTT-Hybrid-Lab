"""F8-R9 read-only storage audit; NEVER a first-use or runner admission.

Tests historical P85 F2 owner route and a *proposed* separate WSL ext4
custody directory. A matching path is only an untrusted local candidate:
owner run identity, source SHA, signed custody, first-use permission, real
fsync/crash results and all science are STILL UNSATISFIED.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FREEZE = HERE / "p85mr_f8r9_wsl_storage_freeze.json"
FREEZE_BLOB = "42957eca523aa33c3d76fbb83ebad8260c55deaa"
ROUTE = PROJECT / ".github/workflows/p85mr-f2-pinned-fitcal-once.yml"
ROUTE_BLOB = "17f80a483eaad4c6942c593df8d8312d82b32c9d"
COLLECTOR = HERE / "p85mr_fitcal_owner_firstuse.py"
COLLECTOR_BLOB = "f5deeed08624d4dbe878a58e6869c9c4bb062ae3"
SCHEMA = "ttt-hybrid.f8r9-readonly-wsl-custody-storage-admission.v1"
DISTRO = "Ubuntu-24.04"
UID = 1000
DENIED_PREFIXES = ("/mnt", "/media", "/run", "/tmp", "/var/tmp",
                   "/dev", "/proc", "/sys")
DENIED_FS = ("9p", "drvfs", "fuse", "fuseblk", "overlay", "tmpfs",
             "nfs", "cifs", "vfat", "ntfs")


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()


def frozen() -> dict:
    raw = FREEZE.read_bytes()
    if _git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R9_FREEZE_BLOB_DRIFT")
    f = json.loads(raw)
    if (f.get("schema") != SCHEMA
            or f.get("stage") != "FROZEN_BEFORE_R9_CODE_OR_TEST"
            or f["base"]["owner_route_blob"] != ROUTE_BLOB
            or f["base"]["owner_collector_blob"] != COLLECTOR_BLOB
            or f["historical_owner_route"]["wsl_distro"] != DISTRO
            or f["historical_owner_route"]["wsl_uid"] != UID
            or f["candidate_storage"]["fs_type"] != "ext4"
            or f["candidate_storage"]["authorized_uid"] != UID
            or f["candidate_storage"]["no_write"] is not True
            or f["authority"]["source_firstuse"] is not False
            or f["authority"]["owner_runner_execution"] is not False):
        raise ValueError("F8R9_SOURCE_OR_AUTHORITY_DRIFT")
    return f


def historical_owner_storage_gap() -> dict:
    """Pin historical one-shot route; it is NOT a live owner assertion."""
    frozen()
    route = ROUTE.read_bytes()
    collector = COLLECTOR.read_bytes()
    if (_git_blob(route) != ROUTE_BLOB
            or _git_blob(collector) != COLLECTOR_BLOB
            or b"wslpath -u $root" not in route
            or b"/mnt/[a-z]" not in route
            or b"TTT-HYBRID-GPU-WIN" not in route
            or b"Ubuntu-24.04" not in route
            or b'base=project/"evidence"/"p85mr_fitcal_firstuse"' not in collector
            or b'attempt=base/"ONCE"' not in collector):
        raise ValueError("F8R9_HISTORICAL_OWNER_OR_STORAGE_DRIFT")
    return {
        "schema": SCHEMA,
        "historical_route_is_pinned": True,
        "original_owner_project_is_drive_mounted": True,
        "historical_evidence_under_project_checkout": True,
        "historical_native_ext4_durable_journal_proven": False,
        "original_F2_approval_transferable_to_F8": False,
        "no_original_source_import_or_replay": True,
        "scientific_pass": False,
    }


def _decode_mount_path(field: str) -> str:
    # mountinfo encodes spaces, tabs, newlines and literal backslash as octal.
    if not field.startswith("/"):
        raise ValueError("F8R9_INVALID_MOUNT_PATH")
    def decode(match):
        return chr(int(match.group(1), 8))
    path = re.sub(r"\\([0-7]{3})", decode, field)
    if ("\\" in path or "\x00" in path or
            not path.startswith("/") or "//" in path or
            any(p in (".", "..") for p in path.split("/"))):
        raise ValueError("F8R9_UNSAFE_MOUNT_ESCAPE")
    return path.rstrip("/") or "/"


def parse_mountinfo(mountinfo: str) -> tuple[tuple[str, str], ...]:
    """Pure, conservative mountinfo reader. Ambiguous mounts fail closed."""
    if (type(mountinfo) is not str or not mountinfo
            or len(mountinfo.encode("utf-8")) > 1024 * 1024):
        raise ValueError("F8R9_INVALID_MOUNTINFO")
    points: dict[str, str] = {}
    for line in mountinfo.splitlines():
        if not line or line.count(" - ") != 1:
            raise ValueError("F8R9_MALFORMED_MOUNTINFO")
        left, right = line.split(" - ")
        fields = left.split()
        tail = right.split()
        if (len(fields) < 6 or len(tail) < 3
                or not fields[0].isdecimal()
                or not fields[1].isdecimal()):
            raise ValueError("F8R9_MALFORMED_MOUNTINFO")
        mount = _decode_mount_path(fields[4])
        if mount in points:
            raise ValueError("F8R9_AMBIGUOUS_MOUNTPOINT")
        fs = tail[0]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", fs):
            raise ValueError("F8R9_INVALID_FS_TYPE")
        points[mount] = fs
    if "/" not in points:
        raise ValueError("F8R9_MISSING_ROOT_MOUNT")
    return tuple(points.items())


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def classify_candidate(*, path: str, username: str, uid: int,
                       mode: int, symlink_free: bool,
                       kernel_release: str, mountinfo: str) -> dict:
    """Pure classification of supplied values; NOT a trusted owner receipt."""
    frozen()
    if (type(username) is not str or
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,32}", username)
            or type(uid) is not int or uid != UID
            or type(mode) is not int or mode != 0o700
            or symlink_free is not True):
        raise PermissionError("F8R9_OWNER_OR_PRIVATE_ACL_UNQUALIFIED")
    wanted = f"/home/{username}/.local/share/ttt-hybrid/f8-custody"
    if (type(path) is not str or path != wanted or
            any(_within(path, prefix) for prefix in DENIED_PREFIXES)
            or type(kernel_release) is not str or
            "microsoft" not in kernel_release.lower()):
        raise PermissionError("F8R9_NOT_NATIVE_WSL_OWNER_HOME")
    mounts = parse_mountinfo(mountinfo)
    matched = [(p, fs) for p, fs in mounts if _within(path, p)]
    if not matched:
        raise PermissionError("F8R9_NO_MOUNT_FOR_CUSTODY")
    matched.sort(key=lambda pair: len(pair[0]), reverse=True)
    mountpoint, fs_type = matched[0]
    if (fs_type != "ext4" or fs_type.lower() in DENIED_FS):
        raise PermissionError("F8R9_DRIVE_OR_UNSUPPORTED_CUSTODY_FS")
    return {
        "schema": SCHEMA, "classification": "LOCAL_VALUES_ONLY",
        "storage_path_candidate": True, "mountpoint": mountpoint,
        "filesystem": fs_type, "expected_owner_uid": UID,
        "source_effects_executed": 0, "owner_run_identity_verified": False,
        "owner_writer_admitted": False, "parent_fsync_proven": False,
        "cross_process_owner_custody_proven": False,
        "source_firstuse_authorized": False, "scientific_pass": False,
    }


def inspect_local_readonly(path: Path) -> dict:
    """No writes or subprocess. Fails on GitHub-hosted/non-WSL systems."""
    if os.name != "posix" or not hasattr(os, "getuid"):
        raise PermissionError("F8R9_NOT_POSIX_WSL")
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("F8R9_ABSOLUTE_PATH_REQUIRED")
    full = str(path)
    if full != path.as_posix() or ".." in path.parts or "." in path.parts:
        raise ValueError("F8R9_UNSAFE_CUSTODY_PATH")
    current = Path("/")
    for part in path.parts[1:]:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("F8R9_SYMLINK_IN_CUSTODY_PATH")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("F8R9_CUSTODY_ROOT_NOT_DIRECTORY")
    if os.getuid() != UID or info.st_uid != UID:
        raise PermissionError("F8R9_WRONG_RUNTIME_OR_STORAGE_OWNER")
    parts = path.parts
    if len(parts) < 3:
        raise PermissionError("F8R9_BAD_OWNER_HOME")
    return classify_candidate(
        path=full, username=parts[2], uid=os.getuid(),
        mode=stat.S_IMODE(info.st_mode), symlink_free=True,
        kernel_release=os.uname().release,
        mountinfo=Path("/proc/self/mountinfo").read_text(encoding="utf-8"),
    )


def refuse_owner_admission(*_args, **_kwargs):
    raise PermissionError("F8R9_OWNER_FS_AND_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
