"""Fresh teacher-only initialization, explicit grid identity and independent recipes."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

from _common import options
from teacher import EDMTeacher, TeacherConfig
from distillation.model import ConsistencyStudent
from distillation.train import parser, train, file_hash
from distillation.dmd import parser as dmd_parser
from experiments.fresh_distillation import foundation_recipe, dmd_recipe, evaluate
from experiments.refinement import parser as evaluation_parser


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);splits=root/'splits';splits.mkdir()
        mapping={'a':0,'b':1,'c':2}
        (splits/'category_to_id.json').write_text(json.dumps(mapping))
        config=TeacherConfig(resolution=8,num_classes=3,model_channels=16,channel_mult=(1,2),
            num_blocks=1,attn_resolutions=(4,),dropout=0)
        teacher=EDMTeacher(config).eval();source=root/'teacher.ckpt'
        torch.save({**teacher.checkpoint(),'category_to_id':mapping,'step':100000},source)
        task=SimpleNamespace(teacher=source,output=root,cache=root/'cache.pt')
        recipe=foundation_recipe(task)
        recipe.update(batch_size=2,microbatch=1,num_workers=0,device='cpu',amp='none',save_every=1)
        recipe.pop('memory_fraction')
        args=parser().parse_args([*options(recipe),'--steps','1','--outdir',str(root/'foundation')])
        assert args.init_student is None and args.resume is None
        assert args.four_sigmas==[80,5,.5,.02]
        fixture=torch.utils.data.TensorDataset(torch.randn(6,3,8,8),torch.tensor([0,1,2,0,1,2]))
        fixture.paths=[f'train/{i}.png' for i in range(6)];fixture.category_ids=[0,1,2,0,1,2]
        construct=ConsistencyStudent.from_teacher
        called=[]
        def fresh(loaded):
            model=construct(loaded)
            for key,value in teacher.backbone.state_dict().items():
                torch.testing.assert_close(model.backbone.state_dict()[key],value,rtol=0,atol=0)
            called.append(True)
            return model
        with patch('distillation.train.PokemonDataset',return_value=fixture),\
             patch('distillation.train.ConsistencyStudent.from_teacher',side_effect=fresh):
            train(args)
        assert called==[True]
        state=torch.load(root/'foundation/training-state.pt',weights_only=True)
        assert state['identity']['teacher_sha256']==file_hash(source)
        assert 'initialization' not in state['identity']
        torch.testing.assert_close(state['target']['four_sigmas'],torch.tensor([80.,5.,.5,.02]))
        bad=copy.deepcopy(args);bad.resume=str(root/'foundation/training-state.pt');bad.steps=2
        bad.four_sigmas=[80,6,.5,.02]
        with patch('distillation.train.PokemonDataset',return_value=fixture):
            try:train(bad)
            except ValueError as error:assert 'Resume' in str(error)
            else:raise AssertionError('Changed grid resume accepted')
        for mode,nfe in [('one',1),('four',4)]:
            branch=dmd_parser().parse_args([*options(dmd_recipe(task,mode,root/'foundation/student.ckpt')),
                '--outdir',str(root/mode)])
            assert branch.sampling_nfe==nfe and branch.fork_from is None and branch.resume is None
            assert branch.generator_freeze_until==100
        task.workdir=task.baseline_root=root
        def evaluate_fixture(command,log):
            # Exercise the real evaluator CLI parser, not a duplicated choices list.
            parsed=evaluation_parser().parse_args([str(value) for value in command[4:]])
            parsed.evaluation_output.mkdir(parents=True)
            (parsed.evaluation_output/'fid.json').write_text(json.dumps({
                'training_reference_fid':30.,'checkpoint_sha256':file_hash(parsed.checkpoint)}))
        with patch('experiments.fresh_distillation.execute',side_effect=evaluate_fixture):
            for mode in ('one','four'):
                row=evaluate(task,root/'foundation/student.ckpt',mode,'check-'+mode)
                assert row['step']==1 and row['fid']==30.
    print('PASS: exact original-teacher backbone initialization; no old student/optimizer; explicit grid/resume guard; fresh independent 1/4 recipes.')


if __name__=='__main__':main()
