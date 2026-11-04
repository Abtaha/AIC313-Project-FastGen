"""End-to-end exclusion and provenance checks for a fresh teacher/student/GAN chain."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from dataset import read_split, category_from_filename
from training.holdout import create, partition, require_same_partition
from training.teacher import parser as teacher_parser, train as train_teacher
from teacher import TeacherConfig
from distillation.train import parser as student_parser, train as train_student, file_hash
from distillation.dmd import parser as dmd_parser, train as train_dmd
from distillation.support import target_cache
from teacher import EDMTeacher


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Expected a partition/provenance rejection')


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'original';source.mkdir()
        mapping={'a':0,'b':1,'c':2}
        paths=[f'PokemonData/{name}/{name}-{i}.png' for name in mapping for i in range(4)]
        (source/'train_split.txt').write_text('\n'.join(paths)+'\n')
        (source/'category_to_id.json').write_text(json.dumps(mapping))
        # No official validation file exists: creating the partition must not read it.
        clean=root/'clean'
        meta=create(source,clean,per_class=1,seed=13)
        assert meta['fit_count']==9 and meta['holdout_count']==3
        assert partition(clean)==meta
        fit=read_split(clean/'train_split.txt');held=read_split(clean/'val_split.txt')
        assert not set(fit)&set(held) and set(fit)|set(held)==set(paths)
        duplicate=root/'same-seed';create(source,duplicate,per_class=1,seed=13)
        assert (clean/'train_split.txt').read_bytes()==(duplicate/'train_split.txt').read_bytes()
        rejects(lambda:require_same_partition(None,meta))
        rejects(lambda:require_same_partition(meta,None))
        require_same_partition(meta,copy.deepcopy(meta))
        accessed=[]
        class Fixture(torch.utils.data.Dataset):
            def __init__(self, rows):
                self.paths=rows
                self.category_ids=[mapping[category_from_filename(p)] for p in rows]
            def __len__(self):return len(self.paths)
            def __getitem__(self,index):
                accessed.append(self.paths[index])
                return torch.zeros(3,8,8),torch.tensor(self.category_ids[index])
        fixture=Fixture(paths)
        config=TeacherConfig(resolution=8,num_classes=3,model_channels=16,channel_mult=(1,2),num_blocks=1,attn_resolutions=(4,),dropout=0)
        args=teacher_parser().parse_args(['--outdir',str(root/'teacher'),'--split-dir',str(clean),
            '--steps','1','--batch-size','2','--microbatch','1','--device','cpu','--amp','none','--num-workers','0',
            '--save-every','1','--snapshot-every','1','--state-snapshot-every','1',
            '--loss-every','1','--fid-every','1','--fid-count','3','--fid-steps','2'])
        seen_indices=[]
        def loss_check(model,dataset,indices,**kwargs):
            seen_indices.append(set(dataset.paths[i] for i in indices))
            return {'0.1':1.,'0.5':1.,'2.0':1.,'10.0':1.}
        with patch('training.teacher.PokemonDataset',return_value=fixture),patch('training.teacher.TeacherConfig',return_value=config),\
             patch('training.teacher.denoising_losses',side_effect=loss_check),patch('training.teacher.holdout_fid',return_value=20.) as fid:
            train_teacher(args)
            assert fid.call_args.kwargs['balanced'] is True
        assert set(accessed)<=set(fit)
        assert seen_indices==[set(fit),set(held)]
        from teacher_monitor import holdout_fid
        from types import SimpleNamespace
        labels_seen=[]
        class Sampler:
            config=SimpleNamespace(num_classes=3,channels=3,resolution=8)
            def eval(self):return self
            def sample(self,shape,**kwargs):
                labels_seen.extend(kwargs['category'].tolist())
                return torch.zeros(shape)
        reference_indices=[i for i,path in enumerate(paths) if path in held]
        def fid_check(generated,reference,**kwargs):
            assert len(list(Path(generated).glob('*.png')))==6
            assert len(list(Path(reference).glob('*.png')))==3
            return 0.
        with patch('cleanfid.fid.compute_fid',side_effect=fid_check):
            value=holdout_fid(Sampler(),fixture,reference_indices,torch.ones(3)/3,
                outdir=root,device='cpu',batch_size=2,count=6,steps=2,balanced=True)
        assert value==0. and labels_seen==[0,0,1,1,2,2]
        for name in ('teacher.ckpt','best-loss.ckpt','best-fid.ckpt','training-state.pt'):
            assert torch.load(root/'teacher'/name,weights_only=True)['data_partition']==meta
        export=root/'teacher/teacher.ckpt'
        student_args=student_parser().parse_args(['--teacher',str(export),'--outdir',str(root/'student'),
            '--split-dir',str(clean),'--steps','1','--batch-size','2','--microbatch','1',
            '--device','cpu','--amp','none','--num-workers','0','--warmup-steps','0'])
        accessed.clear()
        with patch('distillation.train.PokemonDataset',return_value=Fixture(fit)):
            train_student(student_args)
        assert set(accessed)<=set(fit)
        student_path=root/'student/student.ckpt'
        assert torch.load(student_path,weights_only=True)['data_partition']==meta
        contaminated=root/'unknown-parent.ckpt'
        payload=torch.load(student_path,weights_only=True);payload.pop('data_partition');torch.save(payload,contaminated)
        bad=copy.deepcopy(student_args);bad.outdir=str(root/'bad');bad.init_student=str(contaminated)
        rejects(lambda:train_student(bad))
        dmd_args=dmd_parser().parse_args(['--teacher',str(export),'--student',str(student_path),'--cache',str(root/'cache.pt'),
            '--outdir',str(root/'dmd'),'--split-dir',str(clean),'--steps','1','--batch-size','2','--microbatch','1',
            '--device','cpu','--amp','none','--score-warmup','0','--gan-weight','.003','--gan-start','0'])
        dmd_args.teacher_steps,dmd_args.train_per_class=2,1
        teacher=EDMTeacher.load_checkpoint(export)
        target_cache(dmd_args,teacher,file_hash(export),mapping,torch.device('cpu'))
        accessed.clear()
        with patch('distillation.adversarial.PokemonDataset',return_value=Fixture(fit)):
            train_dmd(dmd_args)
        assert set(accessed)<=set(fit)
        assert torch.load(root/'dmd/one_nfe.ckpt',weights_only=True)['data_partition']==meta
        assert torch.load(root/'dmd/initial.ckpt',weights_only=True)['data_partition']==meta
        wrong=copy.deepcopy(dmd_args);wrong.outdir=str(root/'wrong');wrong.split_dir=str(source)
        rejects(lambda:train_dmd(wrong))
        (clean/'train_split.txt').write_text('\n'.join(fit+[held[0]])+'\n')
        rejects(lambda:partition(clean))
        from training.holdout import digest
        meta['fit_sha256']=digest(clean/'train_split.txt');meta['fit_count']+=1
        (clean/'partition.json').write_text(json.dumps(meta))
        rejects(lambda:partition(clean))
    print('PASS: disjoint deterministic split; teacher gradients only on fit; separate holdout monitoring; clean exports; student/GAN exclusion; unknown-parent and changed-partition rejection.')


if __name__=='__main__':main()
