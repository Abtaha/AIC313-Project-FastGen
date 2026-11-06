"""Four-pass backward simulation with DMD/GAN: bounded pure and anchored pilots."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from _common import ROOT, options
from experiments.refinement import digest, execute, evaluation_command, write
from experiments.overnight_one import pin_best
from experiments.overnight_finish import official


def settings(args, regression):
    return dict(teacher=str(args.workdir/'baseline/teacher.ckpt'),
        student=str(args.workdir/'student-followups/few-grid80/snapshots/student-001000.ckpt'),
        cache=str(args.workdir/'one-followups/teacher-cache16.pt'),sampling_nfe=4,
        lr=2e-6,fake_lr=1e-4,batch_size=32,microbatch=4,fake_updates=5,
        guidance=2.,reg_weight=.25,gan_reg_weight=regression,score_warmup=0,
        generator_freeze_until=100,gan_start=0,gan_weight=.003,
        amp='bf16',seed=20261106,save_every=100,log_every=10,
        diagnostic_every=100,diagnostic_count=16,memory_fraction=.38,memory_limit_gb=7.,
        gan_exclude_manifest=str(args.workdir/'monitor.txt'),
        data_root=str(ROOT/'data/pokemon-generation-one-22k'),split_dir=str(ROOT/'data/pokemon-generation-one-22k'))


def main(args):
    root=args.workdir/'multistep-dmd'
    root.mkdir(parents=True,exist_ok=True)
    recipe=settings(args,0.)
    baseline_file=args.workdir/'student-followups/evaluations/few-grid80-1000/fid.json'
    baseline=json.loads(baseline_file.read_text())
    write(root/'plan.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'teacher_sha256':digest(recipe['teacher']),'student_sha256':digest(recipe['student']),
        'recipe':recipe,'arms':{'pure':0.,'anchored':.025},
        'reference':'DMD2 https://arxiv.org/abs/2405.14867 sections 4.3-4.5',
        'method':'Uniformly sampled inference stage; FP32 stopped prefix; DMD and GAN gradients only through selected stage. Fake score learns generated stage mixture.',
        'anchor':'Optional paired teacher multiscale L2 on initial sigma_max denoising only.',
        'warmup':'100 generator-frozen loops with five fake/GAN updates each',
        'max_generator_updates_per_arm':1000,'checks':[350,600,850,1100],
        'safety_stop':'After any check, stop if internal FID > parent + 1.0; from 500 updates stop if > parent + .5.',
        'selection':'Legacy training-only monitor; no official validation-guided selection.',
        'baseline_internal_fid':baseline['training_reference_fid']})
    if args.dry_run:return
    base_row={'step':1000,'fid':baseline['training_reference_fid'],
              'checkpoint':recipe['student'],'checkpoint_sha256':digest(recipe['student'])}
    rows=[]
    for name,regression in [('pure',0.),('anchored',.025)]:
        out=root/name;state=out/'training-state.pt'
        if not (out/'completion.json').exists():
            for step in (100,350,600,850,1100):
                saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
                if saved<step:
                    write(root/'status.json',{'phase':'training','experiment':name,'target_step':step})
                    restore=['--resume',state] if state.exists() else []
                    execute([sys.executable,'-u','-m','distillation.dmd',*options(settings(args,regression)),
                             '--outdir',out,'--steps',step,*restore],root/f'{name}.log')
                if step==100:
                    warm=torch.load(state,map_location='cpu',weights_only=True)
                    if warm['optimizer']['state']:raise RuntimeError('Generator changed during critic warmup')
                    del warm
                    continue
                checkpoint=out/'snapshots'/f'few_nfe-{step:06d}.ckpt'
                destination=root/'evaluations'/f'{name}-{step}'
                write(root/'status.json',{'phase':'evaluating','experiment':name,'step':step})
                execute(evaluation_command(args,checkpoint,'few_nfe',destination),root/'evaluation.log')
                result=json.loads((destination/'fid.json').read_text())
                row={'step':step,'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),
                     'checkpoint_sha256':result['checkpoint_sha256']}
                pin_best(state,row,out)
                if row['fid']>base_row['fid']+(1. if step<600 else .5):
                    write(out/'completion.json',{'phase':'stopped_for_quality',**row})
                    break
            else:write(out/'completion.json',{'phase':'complete',**row})
        rows.append(json.loads((out/'best-state.json').read_text()))
    selected=min([base_row,*rows],key=lambda row:row['fid'])
    write(root/'selected.json',{'selection':'Internal monitoring before official reporting.','best':selected})
    if selected['checkpoint']!=base_row['checkpoint']:
        official(args,root,Path(selected['checkpoint']),'few_nfe','selected-four')
    write(root/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=ROOT/'results/nov03')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();args.workdir=args.workdir.resolve()
    try:main(args)
    except Exception as error:
        write(args.workdir/'multistep-dmd/status.json',{'phase':'failed','error':str(error)})
        raise
