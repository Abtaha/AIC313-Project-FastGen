"""Create and validate a fixed holdout drawn only from the official training split."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import torch

from dataset import category_from_filename, read_split


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def partition(split_dir):
    directory=Path(split_dir)
    path=directory/'partition.json'
    if not path.exists():
        return None
    meta=json.loads(path.read_text())
    if meta.get('format')!='fastgen-clean-holdout-v1':
        raise ValueError('Unknown data partition format')
    for name,key in [('train_split.txt','fit_sha256'),('val_split.txt','holdout_sha256'),
                     ('category_to_id.json','mapping_sha256')]:
        if digest(directory/name)!=meta[key]:
            raise ValueError(f'Partition file changed: {name}')
    source=directory/meta['source_train_manifest']
    if digest(source)!=meta['source_train_sha256']:
        raise ValueError('Source training manifest changed')
    fit,holdout,original=[read_split(p) for p in (directory/'train_split.txt',directory/'val_split.txt',source)]
    if (len(set(fit))!=len(fit) or len(set(holdout))!=len(holdout) or set(fit)&set(holdout)
            or set(fit)|set(holdout)!=set(original)):
        raise ValueError('Partition must be disjoint and cover exactly the permitted training images')
    if len(fit)!=meta['fit_count'] or len(holdout)!=meta['holdout_count']:
        raise ValueError('Partition counts differ')
    categories=set(json.loads((directory/'category_to_id.json').read_text()))
    if any({category_from_filename(p) for p in paths}!=categories for paths in (fit,holdout)):
        raise ValueError('Every class must occur in both partition subsets')
    return meta


def require_same_partition(parent, current):
    """A clean run cannot inherit weights that may have seen the held-out images."""
    if (parent or current) and parent!=current:
        raise ValueError('Data partitions differ: use clean parent weights and the same split-dir')


def create(source_dir, output, *, per_class=10, seed=20261104):
    source_dir,output=Path(source_dir).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory; a fixed partition cannot be overwritten')
    if per_class<1:
        raise ValueError('holdout-per-class must be positive')
    paths=read_split(source_dir/'train_split.txt')
    if len(paths)!=len(set(paths)):
        raise ValueError('Duplicate source training paths')
    groups=defaultdict(list)
    for path in paths:groups[category_from_filename(path)].append(path)
    rng=torch.Generator().manual_seed(seed)
    held=set()
    for name in sorted(groups):
        group=groups[name]
        if len(group)<=per_class:
            raise ValueError(f'Not enough fitting images left in {name}')
        indices=torch.randperm(len(group),generator=rng).tolist()
        held.update(group[i] for i in indices[:per_class])
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train_split.txt',[p for p in paths if p not in held]),
                      ('val_split.txt',[p for p in paths if p in held])]:
        (output/name).write_text('\n'.join(rows)+'\n')
    (output/'category_to_id.json').write_bytes((source_dir/'category_to_id.json').read_bytes())
    meta={'format':'fastgen-clean-holdout-v1','seed':seed,'holdout_per_class':per_class,
          'fit_count':len(paths)-len(held),'holdout_count':len(held),
          'source_train_manifest':os.path.relpath(source_dir/'train_split.txt',output),
          'source_train_sha256':digest(source_dir/'train_split.txt'),
          'fit_sha256':digest(output/'train_split.txt'),'holdout_sha256':digest(output/'val_split.txt'),
          'mapping_sha256':digest(output/'category_to_id.json'),
          'official_validation_used':False,'requires_clean_initialization':True}
    (output/'partition.json').write_text(json.dumps(meta,indent=2)+'\n')
    partition(output)
    return meta


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,default=Path('data/pokemon-generation-one-22k'))
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--holdout-per-class',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261104)
    args=parser.parse_args()
    print(json.dumps(create(args.source_dir,args.outdir,per_class=args.holdout_per_class,seed=args.seed),indent=2))
