"""P77 isolated fresh-seed author-synthetic FIT/VAL selector.

The P69 generator is locked to an immutable Git blob and loaded into a
private module namespace. No source or model in P69/P75 is edited.
This selector never admits P77 OOD or negative cohorts.
"""
import hashlib
import importlib.util
from pathlib import Path

PARENT_BLOB = '51d4c4f16e18e1fb8265a92eee3d5f70b860893b'
SPLIT_SEEDS = {'fit': 977101, 'val': 977103,
               'ood_0': 977107, 'ood_1': 977109, 'ood_2': 977111,
               'negative': 977113}
COUNTS = {'fit': 512, 'val': 192, 'ood_0': 192, 'ood_1': 192,
          'ood_2': 192, 'negative': 96}
SOURCE_NAMESPACE = 'ttt-hybrid/p77/directional-mechanism-v1'
DEV_SPLITS = frozenset(('fit', 'val'))


def _load_parent():
    path = Path(__file__).with_name('p69_causal_shift_source.py')
    raw = path.read_bytes()
    git_blob = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if git_blob != PARENT_BLOB:
        raise ValueError('P77_PARENT_DRIFT')
    spec = importlib.util.spec_from_file_location('_p77_isolated_parent_generator',path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.SEEDS = SPLIT_SEEDS.copy()
    m.COUNTS = COUNTS.copy()
    def source_seed(split, idx):
        message = f'{SOURCE_NAMESPACE}/{SPLIT_SEEDS[split]}/{idx}'.encode()
        return int.from_bytes(hashlib.sha256(message).digest()[:16], 'big')
    m._seed = source_seed
    return m


_parent = _load_parent()


def make_world(split, index):
    if split not in DEV_SPLITS:
        raise ValueError('P77_HOLDOUT_NOT_ADMITTED')
    return _parent.make_world(split,index)


def make_batch(split, count, start=0):
    if split not in DEV_SPLITS:
        raise ValueError('P77_HOLDOUT_NOT_ADMITTED')
    return _parent.make_batch(split,count,start)


def factual(batch):
    return _parent.factual_history(batch)


def audit(batch):
    return _parent.source_integrity(batch)


def digest_world(world):
    return _parent.digest_world(world)
