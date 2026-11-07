"""Four-stage DMD: inference-equivalent prefix, local gradient and strict resume."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from teacher import EDMTeacher, TeacherConfig
from distillation.model import ConsistencyStudent
from distillation.dmd import generate, fresh_score_error, compatible_identity, parser, train
from distillation.support import target_cache
from distillation.train import file_hash


def main():
    torch.set_num_threads(1)
    config=TeacherConfig(resolution=8,num_classes=3,model_channels=16,
        channel_mult=(1,2),num_blocks=1,attn_resolutions=(4,),dropout=0)
    teacher=EDMTeacher(config).eval()
    student=ConsistencyStudent.from_teacher(teacher).eval()
    student.set_four_sigmas([80,5,.5,.02])
    labels=torch.tensor([0,1]);shape=(2,3,8,8)
    legacy=student.sampling_sigmas(1,device='cpu').clone()
    old_state={k:v for k,v in student.state_dict().items() if k!='one_sigma'}
    restored=ConsistencyStudent(config);restored.load_state_dict(old_state)
    torch.testing.assert_close(restored.sampling_sigmas(1,device='cpu'),legacy,rtol=0,atol=0)
    for sigma in (80.,5.,1.):
        student.set_one_sigma(sigma);student.eval()
        seen=[];hook=student.register_forward_pre_hook(lambda _,args:seen.append(float(torch.as_tensor(args[1]))))
        student.sample(shape,device='cpu',category=labels,nfe=1);hook.remove()
        assert seen==[sigma]
        x=generate(student,torch.randn(shape),labels,nfe=1)
        assert torch.isfinite(x).all()
        data=student.export(category_to_id={'a':0,'b':1,'c':2},step=0,guidance=2.)
        clone=ConsistencyStudent(config);clone.load_state_dict({k.removeprefix('backbone.'):v for k,v in data['state_dict'].items()})
        assert clone.one_sigma.item()==sigma
    inputs=[]
    hook=student.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
    torch.manual_seed(17)
    student.sample(shape,device='cpu',category=labels,nfe=4)
    hook.remove();assert len(inputs)==4
    for stage in range(4):
        actual=[];modes=[]
        hook=student.register_forward_pre_hook(lambda module,args:(actual.append(args[0].detach().clone()),modes.append(torch.is_grad_enabled())) and None)
        student.train();student.zero_grad();torch.manual_seed(17)
        output=generate(student,torch.randn(shape),labels,nfe=4,stage=stage)
        hook.remove()
        torch.testing.assert_close(actual[-1],inputs[stage],rtol=0,atol=0)
        assert modes==[False]*stage+[True] and student.training
        output.square().mean().backward()
        assert sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None)>0
    rng=torch.get_rng_state().clone()
    fresh_score_error(student,teacher,count=3,microbatch=2,seed=27,nfe=4)
    assert torch.equal(rng,torch.get_rng_state())
    a={'recipe':{'lr':2e-6}};b={'recipe':{'lr':2e-6,'sampling_nfe':4}}
    assert not compatible_identity(a,b,fork=True)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);mapping={'a':0,'b':1,'c':2}
        (root/'category_to_id.json').write_text(json.dumps(mapping))
        (root/'train_split.txt').write_text('fixture\n')
        teacher_path=root/'teacher.ckpt';initial=root/'student.ckpt'
        torch.save({**teacher.checkpoint(),'category_to_id':mapping,'step':1},teacher_path)
        torch.save(student.export(category_to_id=mapping,step=0,guidance=2.),initial)
        fixture=torch.utils.data.TensorDataset(torch.randn(6,3,8,8),torch.tensor([0,1,2,0,1,2]))
        fixture.paths=[f'train/{i}.png' for i in range(6)];fixture.category_ids=[0,1,2,0,1,2]
        args=parser().parse_args(['--teacher',str(teacher_path),'--student',str(initial),'--cache',str(root/'cache.pt'),
            '--outdir',str(root/'dmd4'),'--steps','1','--batch-size','2','--microbatch','1','--device','cpu',
            '--amp','none','--fake-updates','2','--score-warmup','0','--generator-freeze-until','1','--gan-start','0','--gan-weight','.003',
            '--gan-reg-weight','0','--sampling-nfe','4','--diagnostic-every','1','--diagnostic-count','2'])
        args.teacher_steps,args.train_per_class=2,1
        target_cache(args,teacher,file_hash(teacher_path),mapping,torch.device('cpu'))
        with patch('distillation.adversarial.PokemonDataset',return_value=fixture):
            train(args)
            args.resume=str(root/'dmd4/training-state.pt');args.steps=2
            train(args)
        state=torch.load(root/'dmd4/training-state.pt',weights_only=True)
        assert state['step']==2 and state['identity']['recipe']['sampling_nfe']==4
        assert int(next(iter(state['optimizer']['state'].values()))['step'])==1
        exported=ConsistencyStudent.load_checkpoint(root/'dmd4/few_nfe.ckpt')
        calls=[];hook=exported.register_forward_pre_hook(lambda *args:calls.append(1))
        exported.sample(shape,device='cpu',category=labels,nfe=4);hook.remove()
        assert len(calls)==4
    print('PASS: exact inference prefixes; selected-stage-only gradients; RNG-safe diagnostics; 1/4 identity separation; four-pass GAN warmup/train/resume/export.')


if __name__=='__main__':main()
