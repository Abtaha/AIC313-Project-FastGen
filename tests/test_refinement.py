"""CPU checks for refinement schedules, warmup, provenance and resume behavior."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from teacher import EDMTeacher, TeacherConfig
from distillation.model import ConsistencyStudent
from distillation.dmd import compatible_identity, fresh_score_error, regression_weight
from distillation.dmd import parser as dmd_parser, train as train_dmd
from distillation.support import target_cache
from distillation.train import file_hash
from training.teacher import parser as teacher_parser, train as train_teacher


def main():
    torch.set_num_threads(1)
    config=TeacherConfig(resolution=8,num_classes=3,model_channels=16,channel_mult=(1,2),num_blocks=1,attn_resolutions=(4,),dropout=0)
    teacher=EDMTeacher(config).eval()
    student=ConsistencyStudent.from_teacher(teacher).train()
    fake=EDMTeacher(config).train()
    rng=torch.get_rng_state().clone()
    result=fresh_score_error(student,fake,count=3,microbatch=2,seed=123)
    assert set(result)=={'0.1','0.5','2.0','10.0'}
    assert torch.equal(rng,torch.get_rng_state())
    assert student.training and fake.training
    args=dmd_parser().parse_args(['--teacher','a','--student','b','--cache','c','--outdir','d',
        '--reg-decay-start','4','--reg-decay-end','6','--gan-reg-weight','.025'])
    assert [regression_weight(args,step,True) for step in (3,4,5,6,7)]==[.025,.025,.0125,0,0]
    original={'recipe':{'fake_updates':3,'lr':5e-6},'teacher_sha256':'fixed'}
    changed=copy.deepcopy(original)
    changed['recipe'].update(fake_updates=5,reg_decay_start=4,reg_decay_end=6,generator_freeze_until=4)
    changed['gan_excluded_manifest_sha256']='monitor'
    assert compatible_identity(original,changed,fork=True)
    assert not compatible_identity(original,changed)
    cache_change=copy.deepcopy(original)
    original['cache_sha256']='old'
    cache_change['cache_sha256']='new'
    assert not compatible_identity(original,cache_change,fork=True)
    assert compatible_identity(original,cache_change,fork=True,refresh_cache=True)
    cache_change['teacher_sha256']='different'
    assert not compatible_identity(original,cache_change,fork=True,refresh_cache=True)
    changed['cache_sha256']='old'
    changed['recipe']['lr']=1e-4
    assert not compatible_identity(original,changed,fork=True)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); mapping={'a':0,'b':1,'c':2}
        base=root/'teacher.ckpt'
        torch.save({**teacher.checkpoint(),'category_to_id':mapping,'step':10,'seen_images':1280},base)
        (root/'category_to_id.json').write_text(json.dumps(mapping))
        (root/'train_split.txt').write_text('train-only fixture\n')
        fixture=torch.utils.data.TensorDataset(torch.randn(6,3,8,8),torch.tensor([0,1,2,0,1,2]))
        fixture.paths=[f'train/{i}.png' for i in range(6)]
        fixture.category_ids=[0,1,2,0,1,2]
        args=teacher_parser().parse_args(['--init-teacher',str(base),'--steps','11','--outdir',str(root/'teacher'),
            '--batch-size','2','--microbatch','1','--device','cpu','--amp','none','--split-dir',str(root),
            '--num-workers','0','--loss-every','0','--fid-every','0','--lr','2e-5','--lr-rampup-kimg','0'])
        with patch('training.teacher.PokemonDataset',return_value=fixture),patch('training.teacher.TeacherConfig',return_value=config):
            train_teacher(args)
            args.init_teacher=None
            args.resume=str(root/'teacher/training-state.pt');args.steps=12
            train_teacher(args)
        state=torch.load(root/'teacher/training-state.pt',weights_only=True)
        assert state['step']==12 and state['initialization']['step']==10
        assert state['optimizer']['param_groups'][0]['lr']==2e-5
        student_path=root/'student.ckpt'
        torch.save(student.export(category_to_id=mapping,step=0,guidance=2.),student_path)
        args=dmd_parser().parse_args(['--teacher',str(base),'--student',str(student_path),'--cache',str(root/'cache.pt'),
            '--outdir',str(root/'dmd'),'--steps','1','--batch-size','2','--microbatch','1','--device','cpu',
            '--amp','none','--score-warmup','0','--fake-updates','5','--gan-weight','.003','--gan-start','0',
            '--generator-freeze-until','1','--reg-decay-start','1','--reg-decay-end','2',
            '--diagnostic-every','1','--diagnostic-count','2'])
        args.teacher_steps,args.train_per_class=2,2
        target_cache(args,teacher,file_hash(base),mapping,torch.device('cpu'))
        with patch('distillation.adversarial.PokemonDataset',return_value=fixture):
            train_dmd(args)
            first=torch.load(root/'dmd/training-state.pt',weights_only=True)
            for key,value in student.state_dict().items():
                torch.testing.assert_close(first['generator'][key],value)
            assert not first['optimizer']['state']
            boost=copy.deepcopy(args)
            boost.outdir=str(root/'boost-warmup')
            boost.generator_gan_weight=.01
            train_dmd(boost)
            boosted=torch.load(root/'boost-warmup/training-state.pt',weights_only=True)
            for module in ('fake','gan_head'):
                assert all(torch.equal(first[module][key],value) for key,value in boosted[module].items())
            boost.resume=str(root/'boost-warmup/training-state.pt');boost.steps=2
            train_dmd(boost)
            boost_state=torch.load(root/'boost-warmup/training-state.pt',weights_only=True)
            assert int(next(iter(boost_state['optimizer']['state'].values()))['step'])==1
            args.resume=str(root/'dmd/training-state.pt');args.steps=2
            train_dmd(args)
        final=torch.load(root/'dmd/training-state.pt',weights_only=True)
        assert int(next(iter(final['fake_optimizer']['state'].values()))['step'])==10
        assert int(next(iter(final['optimizer']['state'].values()))['step'])==1
        rows=[json.loads(line) for line in (root/'dmd/stats.jsonl').read_text().splitlines()]
        assert [row['reg_weight'] for row in rows]==[.025,0.]
        assert [row['generator_updated'] for row in rows]==[False,True]
        assert len((root/'dmd/diagnostics.jsonl').read_text().splitlines())==2
        from distillation.adversarial import BalancedTrainingImages
        excluded=root/'excluded.txt';excluded.write_text(fixture.paths[0]+'\n')
        with patch('distillation.adversarial.PokemonDataset',return_value=fixture):
            images=BalancedTrainingImages(root,root,mapping,excluded)
        assert images.training_count==5 and 0 not in images.indices[0].tolist()
    print('PASS: teacher EMA initialization/resume; strict fork identity; regression decay to zero; score-only warmup; fresh diagnostics preserve RNG; monitoring exclusion.')


if __name__=='__main__':main()
