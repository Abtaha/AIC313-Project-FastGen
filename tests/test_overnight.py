"""Verify that overnight retention pins the evaluated atomic full state, not a later save."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from experiments.overnight_one import pin_best, wait_for_cache16


def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        source=root/'state.pt';board=root/'best'
        torch.save({'step':5250,'generator':torch.tensor([1.])},source)
        first={'step':5250,'fid':40.1,'checkpoint':'first.ckpt','checkpoint_sha256':'a'}
        pin_best(source,first,board)
        old_inode=(board/'best-training-state.pt').stat().st_ino
        temporary=root/'next.pt'
        torch.save({'step':5500,'generator':torch.tensor([2.])},temporary)
        temporary.replace(source)
        assert torch.load(board/'best-training-state.pt',weights_only=True)['step']==5250
        # A better score referring to the old step cannot pin the new source inode.
        pin_best(source,{**first,'fid':39.},board)
        assert (board/'best-training-state.pt').stat().st_ino==old_inode
        second={**first,'step':5500,'fid':39.5}
        pin_best(source,second,board)
        assert torch.load(board/'best-training-state.pt',weights_only=True)['step']==5500
        pin_best(source,{**second,'fid':42.},board)
        assert json.loads((board/'best-state.json').read_text())['fid']==39.5
        work=root/'experiment';upstream=work/'one-followups'
        (upstream/'cache16').mkdir(parents=True)
        torch.save({'step':6100},upstream/'cache16/training-state.pt')
        (upstream/'cache16-status.json').write_text(json.dumps({'phase':'complete'}))
        scores=upstream/'evaluations/cache16-6100';scores.mkdir(parents=True)
        (scores/'fid.json').write_text(json.dumps({'training_reference_fid':39.,'checkpoint':'last.ckpt','checkpoint_sha256':'b'}))
        overnight=work/'overnight-one';overnight.mkdir()
        wait_for_cache16(SimpleNamespace(workdir=work),overnight)
        assert json.loads((overnight/'cache16-best/best-state.json').read_text())['step']==6100
    print('PASS: atomic full-state retention, step/score matching, best-only replacement, and upstream completion gate.')


if __name__=='__main__':main()
