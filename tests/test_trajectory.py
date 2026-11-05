"""Verify inference-prefix equivalence, detached rollout, and training/resume."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from teacher import EDMTeacher, TeacherConfig
from distillation.model import ConsistencyStudent
from distillation.loss import consistency_loss, noise_grid, trajectory_input
from distillation.train import parser, train


def main():
    torch.set_num_threads(1)
    config=TeacherConfig(resolution=8,num_classes=3,model_channels=16,
        channel_mult=(1,2),num_blocks=1,attn_resolutions=(4,),dropout=0)
    teacher=EDMTeacher(config).eval().requires_grad_(False)
    student=ConsistencyStudent.from_teacher(teacher)
    student.set_four_sigmas([80,5,.5,.02])
    images=torch.randn(2,3,8,8)
    labels=torch.tensor([0,1])
    grid=noise_grid(config,80,'cpu')
    # Capture all actual inference inputs, including un-clipped intermediate outputs.
    captured=[]
    hook=student.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    student.eval();torch.manual_seed(17)
    student.sample(images.shape,device='cpu',category=labels,nfe=4)
    hook.remove()
    assert len(captured)==4
    for stage in range(4):
        student.train();torch.manual_seed(17)
        x,low,high=trajectory_input(student,images,labels,grid,stage=stage)
        torch.testing.assert_close(x,captured[stage],rtol=0,atol=0)
        assert student.training and not x.requires_grad
        assert torch.all(low<high) and torch.all(low>=config.sigma_min)
    target=copy.deepcopy(student).eval().requires_grad_(False)
    student.zero_grad()
    torch.manual_seed(19)
    loss=consistency_loss(student,target,teacher,images,labels,grid=grid,guidance=2.,trajectory=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in target.parameters())
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);mapping={'a':0,'b':1,'c':2}
        (root/'category_to_id.json').write_text(json.dumps(mapping))
        base=root/'teacher.ckpt'
        torch.save({**teacher.checkpoint(),'category_to_id':mapping,'step':1},base)
        initial=root/'initial.ckpt'
        torch.save(student.export(category_to_id=mapping,step=0,guidance=2.),initial)
        fixture=torch.utils.data.TensorDataset(torch.randn(6,3,8,8),torch.tensor([0,1,2,0,1,2]))
        fixture.paths=[f'train/{i}.png' for i in range(6)]
        fixture.category_ids=[0,1,2,0,1,2]
        for source in ('real','student'):
            out=root/source
            args=parser().parse_args(['--teacher',str(base),'--init-student',str(initial),'--outdir',str(out),
                '--split-dir',str(root),'--steps','1','--batch-size','2','--microbatch','1',
                '--num-workers','0','--device','cpu','--amp','none','--num-intervals','80',
                '--trajectory-probability','1','--trajectory-source',source,'--save-every','1'])
            with patch('distillation.train.PokemonDataset',return_value=fixture):
                train(args)
                args.resume=str(out/'training-state.pt');args.steps=2
                train(args)
                args.trajectory_source='real' if source=='student' else 'student'
                args.steps=3
                try:train(args)
                except ValueError as error:assert 'Resume' in str(error)
                else:raise AssertionError('Objective-changing resume accepted')
            state=torch.load(out/'training-state.pt',weights_only=True)
            assert state['step']==2 and state['identity']['recipe']['trajectory_source']==source
            rows=[json.loads(s) for s in (out/'stats.jsonl').read_text().splitlines()]
            assert rows[0]['trajectory_batches']==2
    print('PASS: exact four-pass inference prefixes; stopped rollout; finite nonzero gradients; frozen targets; both arms train/resume; changed objective rejected.')


if __name__=='__main__':main()
