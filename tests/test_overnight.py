"""Verify that overnight retention pins the evaluated atomic full state, not a later save."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from experiments.overnight_one import pin_best, wait_for_cache16
from experiments.student_followups import few_recipe, one_recipe, extend_four


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
        args=SimpleNamespace(workdir=work,dry_run=False)
        control=few_recipe(args,{})
        for change in ({'num_intervals':80},{'ema_decay':.995}):
            arm=few_recipe(args,change)
            assert {key for key in arm if arm[key]!=control[key]}==set(change)
        with patch('experiments.student_followups.one_settings',return_value={'lr':5e-6,'gan_reg_weight':.025}):
            low,high=one_recipe(args,.05),one_recipe(args,.1)
            assert {key for key in low if low[key]!=high[key]}=={'gan_reg_weight'}
            assert low['fork_refresh_cache'] and low['lr']==5e-6
        parent=work/'student-followups/few-grid80';parent.mkdir(parents=True)
        (parent.parent/'status.json').write_text(json.dumps({'phase':'complete'}))
        original={'step':1000,'optimizer':{'momentum':torch.tensor([7.])}}
        torch.save(original,parent/'best-training-state.pt')
        (parent/'best-state.json').write_text(json.dumps({'step':1000,'fid':30.5,
            'checkpoint':'protected.ckpt','checkpoint_sha256':'original'}))
        with patch('experiments.student_followups.few_blocks') as run:
            extend_four(args)
        run.assert_called_once()
        assert run.call_args.kwargs['targets']==(1250,1500,1750,2000)
        assert run.call_args.kwargs['fixed_limit']==31.25
        resumed=torch.load(work/'four-extension/grid80/training-state.pt',weights_only=True)
        assert resumed['step']==1000 and resumed['optimizer']['momentum'].item()==7.
        assert json.loads((work/'four-extension/grid80/best-state.json').read_text())['step']==1000
        # Full overnight orchestration must hand each fresh DMD fork an empty outdir.
        from experiments.overnight_finish import main as finish
        parent=work/'student-followups/anchor05';parent.mkdir()
        torch.save({'step':5500},parent/'best-training-state.pt')
        (parent/'best-state.json').write_text(json.dumps({'step':5500,'fid':39.9,
            'checkpoint':'one-parent.ckpt','checkpoint_sha256':'one'}))
        (work/'trajectory-four').mkdir()
        (work/'trajectory-four/status.json').write_text(json.dumps({'phase':'complete'}))
        (work/'four-extension/grid80/best-state.json').write_text(json.dumps({'step':2000,'fid':30.45,
            'checkpoint':'four.ckpt','checkpoint_sha256':'four'}))
        teacher=root/'runs/holdout-v1/teacher/snapshots';teacher.mkdir(parents=True)
        (teacher/'teacher-0100000.ckpt').write_text('fixture')
        board=work/'overnight-finish/four-extra';board.mkdir(parents=True)
        (board/'best-state.json').write_text(json.dumps({'step':2500,'fid':20.,
            'checkpoint':'four-extra.ckpt','checkpoint_sha256':'four-extra'}))
        def launch(args,queue,name,settings,source,targets,limit):
            out=queue/name
            assert not out.exists(), 'Fresh trainer output was populated before launch'
            out.mkdir()
            row={'step':5750,'fid':39.8 if name=='control' else 39.,
                 'checkpoint':name+'.ckpt','checkpoint_sha256':name}
            (out/'best-state.json').write_text(json.dumps(row))
        with patch('experiments.overnight_finish.ROOT',root),\
             patch('experiments.overnight_finish.subprocess.check_output',return_value='test'),\
             patch('experiments.overnight_finish.one_settings',return_value={}),\
             patch('experiments.overnight_finish.blocks',side_effect=launch),\
             patch('experiments.overnight_finish.official') as report:
            finish(args)
        assert report.call_args_list[0].args[2]==Path('teacher-features.ckpt')
        assert report.call_args_list[0].args[3]=='one_nfe'
        assert report.call_args_list[1].args[2]==Path('four-extra.ckpt')
    print('PASS: atomic full-state retention, step/score matching, best-only replacement, and upstream completion gate.')


if __name__=='__main__':main()
