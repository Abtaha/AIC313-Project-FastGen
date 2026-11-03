"""Verify packaged checkpoint hashes, protected code, parameter caps, and actual NFE."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

from model import Model
from src.utils import count_parameters

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'SUBMISSION_MANIFEST.json').read_text())
    for name, digest in manifest['protected_files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    base = next(node for node in ast.parse((ROOT/'model.py').read_text()).body
                if isinstance(node,ast.ClassDef) and node.name=='Model')
    digest = hashlib.sha256(ast.dump(base,include_attributes=False).encode()).hexdigest()
    assert digest == manifest['official_base_model_ast_sha256'], 'Base Model class differs'
    rows=[]
    for mode,nfe in [('one_nfe',1),('few_nfe',4)]:
        source=ROOT/'checkpoints'/(mode+'.ckpt')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['checkpoints'][mode]['sha256']
        network=Model.load_checkpoint(str(source),evaluate_mode=mode,device=args.device)
        assert 0 < count_parameters(network,trainable_only=False) <= 100_000_000
        calls=[]
        hook=network.backbone.backbone.register_forward_hook(lambda *args:calls.append(1))
        torch.manual_seed(2026)
        with torch.no_grad():
            images=network.sample((2,3,64,64),device=args.device,
                                  category=torch.tensor([0,150],device=args.device))
        hook.remove()
        assert len(calls)==nfe,(mode,len(calls))
        assert images.shape==(2,3,64,64) and torch.isfinite(images).all()
        assert images.min()>=-1 and images.max()<=1
        row={'mode':mode,'parameters':count_parameters(network,trainable_only=False),
             'actual_backbone_calls':len(calls),'shape':list(images.shape),'finite':True}
        print(json.dumps(row),flush=True)
        rows.append(row)
        del network
    print('PASS: official files/base class unchanged; both checkpoints load at required paths; NFE, shapes, ranges, and caps verified.')


if __name__=='__main__':
    main()
