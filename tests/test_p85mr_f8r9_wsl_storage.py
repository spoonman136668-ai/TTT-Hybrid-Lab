"""F8-R9 isolated mount/source-path admission negatives, no owner actions."""
from pathlib import Path
import inspect
import pytest

from experiments import p85mr_f8r9_wsl_storage as s

EXT4 = ("1 0 8:1 / / rw,relatime - ext4 /dev/sdb rw\n"
        "2 1 0:45 / /mnt/c rw,relatime - 9p drvfs rw\n")
KERNEL = "6.6.87.2-microsoft-standard-WSL2"


def candidate(**changes):
    options = dict(path="/home/research/.local/share/ttt-hybrid/f8-custody",
        username="research", uid=1000, mode=0o700, symlink_free=True,
        kernel_release=KERNEL, mountinfo=EXT4)
    options.update(changes)
    return s.classify_candidate(**options)


def test_exact_source_freeze_and_historical_owner_drive_mount_gap():
    freeze = s.frozen()
    assert freeze["historical_owner_route"]["runner"] == "TTT-HYBRID-GPU-WIN"
    assert freeze["authority"]["source_firstuse"] is False
    audit = s.historical_owner_storage_gap()
    assert audit["historical_route_is_pinned"]
    assert audit["original_owner_project_is_drive_mounted"]
    assert not audit["historical_native_ext4_durable_journal_proven"]
    assert not audit["original_F2_approval_transferable_to_F8"]


def test_only_linux_native_wsl_ext4_private_owner_home_candidate():
    x = candidate()
    assert x["storage_path_candidate"] is True
    assert x["filesystem"] == "ext4"
    assert x["owner_writer_admitted"] is False
    assert x["parent_fsync_proven"] is False
    assert x["source_firstuse_authorized"] is False
    assert x["scientific_pass"] is False
    with pytest.raises(PermissionError, match="NOT_AUTHORIZED"):
        s.refuse_owner_admission()


@pytest.mark.parametrize("wrong", [
    {"path":"/mnt/c/Users/research/f8-custody"},
    {"path":"/home/research/.local/share/ttt-hybrid/f8-custody/../new"},
    {"path":"/tmp/f8-custody"},
    {"kernel_release":"6.10.0-generic"},
    {"uid":0},
    {"uid":True},
    {"username":"root/../x"},
    {"mode":0o755},
    {"mode":0o770},
    {"mode":True},
    {"symlink_free":False},
])
def test_reject_unsafe_owner_or_host(wrong):
    with pytest.raises((ValueError, PermissionError)):
        candidate(**wrong)


@pytest.mark.parametrize("bad_fs", ["9p", "drvfs", "overlay",
                                    "tmpfs", "nfs", "cifs"])
def test_reject_journal_overmounted_by_non_native_source(bad_fs):
    mountinfo = EXT4 + (
        f"30 1 0:99 / /home/research/.local/share/ttt-hybrid "
        f"rw,relatime - {bad_fs} origin rw\n")
    with pytest.raises(PermissionError, match="DRIVE_OR_UNSUPPORTED"):
        candidate(mountinfo=mountinfo)


def test_longest_mount_resolution_and_mount_escape():
    clean = EXT4 + ("40 1 8:2 / /home/research/.local/share "
                    "rw - ext4 /dev/sdc rw\n")
    assert candidate(mountinfo=clean)["mountpoint"] == "/home/research/.local/share"
    escaped = "1 0 8:1 / / rw - ext4 /dev/sdb rw\n2 1 8:3 / /home/foo\\040bar rw - ext4 /dev/sdc rw\n"
    assert ("foo bar", "ext4") not in s.parse_mountinfo(escaped)
    assert ("/home/foo bar", "ext4") in s.parse_mountinfo(escaped)


@pytest.mark.parametrize("broken", [
    "", "x", "1 0 8:1 / / rw - ext4 /dev/sda rw\n"
    "2 1 8:2 / / rw - ext4 /dev/sdb rw\n",
    "1 0 8:1 / / rw - ext4 /dev/sda rw\n"
    "2 1 8:2 / /a\\09Z rw - ext4 /dev/sdb rw\n",
    "2 0 8:1 /mnt /mnt rw - 9p drvfs rw\n",
])
def test_bad_or_ambiguous_mountinfo_fails_closed(broken):
    with pytest.raises(ValueError):
        s.parse_mountinfo(broken)


def test_readonly_runtime_does_not_forge_WSL_identity_on_host(tmp_path):
    # A Linux GitHub Actions runner is not the TTT owner WSL instance.
    with pytest.raises((PermissionError, ValueError, FileNotFoundError)):
        s.inspect_local_readonly(tmp_path)


def test_no_local_mutation_or_source_dispatch():
    x = inspect.getsource(s)
    for forbidden in ("os.mkdir(", "mkdir(", "os.open(", "write_text(",
                      "write_bytes(", "env.reset(", "env.do(",
                      "subprocess.run(", "workflow_dispatch(",
                      "torch.save(", "requests.get(", "urlopen("):
        assert forbidden not in x
    assert s.frozen()["authority"]["owner_runner_execution"] is False
