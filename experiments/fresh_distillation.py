"""Original-teacher-only fresh warmstart followed by independent 1/4-NFE DMD."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from _common import ROOT, options
from experiments.refinement import digest, execute, evaluation_command, write
from experiments.overnight_one import pin_best
from experiments.overnight_finish import official

ORIGINAL_TEACHER='b108931ca379200ca17f3449cc50e186263a2d71913efebce6b002e30e1bbf9c'


def foundation_recipe(args):
    return dict(teacher=str(args.teacher),data_root=str(ROOT/'data/pokemon-generation-one-22k'),
        split_dir=str(args.output/'splits'),batch_size=128,microbatch=16,num_workers=2,
        num_intervals=80,four_sigmas=[80,5,.5,.02],guidance=2.,lr=1e-4,warmup_steps=500,
        ema_decay=.95,loss='pseudo-huber',grad_clip=1.,amp='bf16',hflip=True,seed=42,
        save_every=1000,sample_every=1000,log_every=50,memory_fraction=.27,memory_limit_gb=5.5)


def dmd_recipe(args, mode, student):
    return dict(teacher=str(args.teacher),student=str(student),cache=str(args.cache),
        data_root=str(ROOT/'data/pokemon-generation-one-22k'),split_dir=str(args.output/'splits'),
        sampling_nfe=1 if mode=='one' else 4,batch_size=32,microbatch=4,
        lr=5e-6 if mode=='one' else 2e-6,fake_lr=1e-4,fake_updates=5,
        guidance=2.,reg_weight=.25,gan_reg_weight=.05 if mode=='one' else 0.,
        score_warmup=0,generator_freeze_until=100,gan_start=0,gan_weight=.003,
        amp='bf16',seed=789,save_every=1000000,log_every=25,
        diagnostic_every=500,diagnostic_count=16,memory_fraction=.38,memory_limit_gb=7.)


def evaluate(args, checkpoint, mode, name):
    destination=args.output/'evaluations'/name
    kind='student' if mode=='one' else 'few_nfe'
    execute(evaluation_command(args,checkpoint,kind,destination),args.output/f'{mode}-evaluation.log')
    result=json.loads((destination/'fid.json').read_text())
    return {'step':torch.load(checkpoint,map_location='cpu',weights_only=True)['step'],
        'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),
        'checkpoint_sha256':result['checkpoint_sha256']}


def prepare(args):
    if digest(args.teacher)!=ORIGINAL_TEACHER:
        raise ValueError('This experiment requires the original 20.18-FID teacher')
    split=ROOT/'data/pokemon-generation-one-22k'
    permitted=(split/'train_split.txt').read_text().splitlines()
    held=set((args.workdir/'monitor.txt').read_text().splitlines())
    if not held<=set(permitted):raise ValueError('Monitor is not a subset of permitted training data')
    fitting=[p for p in permitted if p not in held]
    folder=args.output/'splits';folder.mkdir(parents=True,exist_ok=True)
    content='\n'.join(fitting)+'\n'
    if (folder/'train_split.txt').exists() and (folder/'train_split.txt').read_text()!=content:
        raise ValueError('Fitting manifest changed')
    (folder/'train_split.txt').write_text(content)
    (folder/'category_to_id.json').write_bytes((split/'category_to_id.json').read_bytes())
    cache=torch.load(args.cache,map_location='cpu',weights_only=True)
    if cache['identity']['teacher_sha256']!=ORIGINAL_TEACHER or cache['identity']['guidance']!=2.:
        raise ValueError('Synthetic cache must belong to the original guided teacher')
    del cache
    write(args.output/'plan.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'teacher_sha256':ORIGINAL_TEACHER,'existing_student_weights_used':False,
        'initialization':'Original teacher backbone; no old student or optimizer loaded.',
        'fitting_images':len(fitting),'monitor_images':len(held),
        'monitor_caveat':'Excluded from all fresh student/GAN fitting; original teacher may have seen them.',
        'official_validation_used_for_selection':False,'cache_sha256':digest(args.cache),
        'foundation':foundation_recipe(args),'foundation_steps':6000,
        'branches':{'one_max_generator_updates':3000,'four_max_generator_updates':1500},
        'branch_initialization':'Internally selected fresh foundation export; fresh generator/fake/GAN optimizer states.',
        'selection':'Training-only monitor; separately report official-reference FID.'})


def foundation(args):
    out=args.output/'foundation';state=out/'training-state.pt'
    choices={'one':[],'four':[]}
    for step in (2000,4000,6000):
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(args.output/'status.json',{'phase':'foundation','target_step':step})
            restore=['--resume',state] if state.exists() else []
            execute([sys.executable,'-u','-m','distillation.train',*options(foundation_recipe(args)),
                     '--outdir',out,'--steps',step,*restore],args.output/'foundation.log')
        checkpoint=out/'snapshots'/f'student-{step:06d}.ckpt'
        for mode in choices:
            choices[mode].append(evaluate(args,checkpoint,mode,f'foundation-{mode}-{step}'))
        write(args.output/'foundation-selection.json',{mode:min(rows,key=lambda row:row['fid'])
              for mode,rows in choices.items()})


def branch(args, mode):
    out=args.output/mode;state=out/'training-state.pt'
    parent=json.loads((args.output/'foundation-selection.json').read_text())[mode]
    recipe=dmd_recipe(args,mode,parent['checkpoint'])
    targets=(100,600,1100,1600,2100,2600,3100) if mode=='one' else (100,600,1100,1600)
    rows=[parent]
    for step in targets:
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(args.output/f'{mode}-status.json',{'phase':'training','target_step':step})
            restore=['--resume',state] if state.exists() else []
            execute([sys.executable,'-u','-m','distillation.dmd',*options(recipe),
                     '--outdir',out,'--steps',step,*restore],args.output/f'{mode}.log')
        if step==100:continue
        name='one_nfe' if mode=='one' else 'few_nfe'
        checkpoint=out/'snapshots'/f'{name}-{step:06d}.ckpt'
        row=evaluate(args,checkpoint,mode,f'{mode}-{step}')
        previous=min(rows,key=lambda item:item['fid'])
        pin_best(state,row,out);rows.append(row)
        write(args.output/f'{mode}-history.json',rows)
        if row['fid']>previous['fid']+2:
            break
    selected=min(rows,key=lambda row:row['fid'])
    write(args.output/f'{mode}-selected.json',selected)
    official(args,args.output,Path(selected['checkpoint']),'one_nfe' if mode=='one' else 'few_nfe',f'fresh-{mode}')
    write(args.output/f'{mode}-status.json',{'phase':'complete'})


def main(args):
    if args.phase in ('one','four'):
        if digest(args.teacher)!=ORIGINAL_TEACHER:
            raise ValueError('Branch requires the original teacher')
        branch(args,args.phase);return
    prepare(args)
    if args.dry_run:return
    foundation(args)
    # Old jobs have a finite budget. Avoid overlapping four independent DMD trainers.
    for name in ('overnight-finish','multistep-dmd'):
        path=args.workdir/name/'status.json'
        while path.exists() and json.loads(path.read_text())['phase'] not in ('complete','failed'):
            write(args.output/'status.json',{'phase':'waiting_for_existing_jobs','upstream':name})
            time.sleep(30)
    jobs=[]
    for mode in ('one','four'):
        command=[sys.executable,'-u','-m','experiments.fresh_distillation','--phase',mode,
            '--teacher',str(args.teacher),'--cache',str(args.cache),'--workdir',str(args.workdir),
            '--output',str(args.output),'--baseline-root',str(args.baseline_root)]
        stream=(args.output/f'{mode}-runner.log').open('a')
        jobs.append((mode,subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT),stream))
    write(args.output/'status.json',{'phase':'independent_one_and_four_branches'})
    results={}
    for mode,job,stream in jobs:
        results[mode]=job.wait();stream.close()
    if any(results.values()):raise RuntimeError(f'Branch failures: {results}')
    write(args.output/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=ROOT/'results/nov03')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/fresh-distillation-original')
    parser.add_argument('--phase',choices=('all','one','four'),default='all')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    for key in ('teacher','cache','workdir','output','baseline_root'):setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    try:main(args)
    except Exception as error:
        name='status.json' if args.phase=='all' else f'{args.phase}-status.json'
        write(args.output/name,{'phase':'failed','error':str(error)})
        raise
