"""P75 source selector: frozen P69 causal generator, fresh P75-only seed domain.

The source generator is independently loaded into a private module, never
patched in place. The original source Git blob is locked before use. Tests
may inspect tiny P75 FIT/VAL engineering samples, but no OOD evaluation or
scientific first use is authorized by importing this module.
"""
import hashlib
import importlib.util
from pathlib import Path

SOURCE_BLOB = '51d4c4f16e18e1fb8265a92eee3d5f70b860893b'
SEEDS = {'fit':975101,'val':975103,'ood_0':975107,
         'ood_1':975109,'ood_2':975111,'negative':975113}
COUNTS = {'fit':512,'val':192,'ood_0':192,
          'ood_1':192,'ood_2':192,'negative':96}
P75_NAMESPACE = 'ttt-hybrid/p75/prospective-v1'


def _locked_generator():
    path = Path(__file__).with_name('p69_causal_shift_source.py')
    data = path.read_bytes()
    blob = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
    if blob != SOURCE_BLOB:
        raise ValueError('P75_PARENT_SOURCE_BLOB_DRIFT')
    spec=importlib.util.spec_from_file_location('_p75_isolated_p69_generator',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEEDS=SEEDS.copy()
    module.COUNTS=COUNTS.copy()
    def isolated_seed(split,index):
        payload=f'{P75_NAMESPACE}/{SEEDS[split]}/{index}'.encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:16],'big')
    module._seed=isolated_seed
    return module


_source = _locked_generator()


DEV_SPLITS = frozenset(('fit','val'))


def make_world(split,index):
    if split not in DEV_SPLITS:
        raise ValueError('P75_UNADMITTED_SOURCE')
    return _source.make_world(split,index)


def make_batch(split,count,start=0):
    if split not in DEV_SPLITS:
        raise ValueError('P75_UNADMITTED_SOURCE')
    return _source.make_batch(split,count,start)


def factual_history(world_batch):
    return _source.factual_history(world_batch)


def source_integrity(world_batch):
    return _source.source_integrity(world_batch)


def digest_world(world):
    return _source.digest_world(world)
