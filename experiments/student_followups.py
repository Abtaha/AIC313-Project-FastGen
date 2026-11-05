"""Controlled one/four-pass pilots after the larger teacher-cache experiments."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from _common import ROOT, options
from experiments.overnight_one import blocks, pin_best
from experiments.refinement import digest, evaluation_command, execute, one_settings, write


def one_recipe(args, weight):
    return {**one_settings(args),
            'cache':str(args.workdir/'one-followups/teacher-cache16.pt'),
            'fork_refresh_cache':True, 'gan_reg_weight':weight}


def few_recipe(args, change):
    recipes=json.loads((ROOT/'recipes.json').read_text())
    return {**recipes['consistency'], **recipes['refinement'],
            'lr':5e-6, 'microbatch':16, 'save_every':250, 'sample_every':250,
            'log_every':25, 'memory_limit_gb':7.,
            'teacher':str(args.workdir/'baseline/teacher.ckpt'),
            'init_student':str(args.workdir/'baseline/few_nfe.ckpt'),
            'data_root':str(ROOT/'data/pokemon-generation-one-22k'),
            'split_dir':str(ROOT/'data/pokemon-generation-one-22k'), **change}


def few_blocks(args, root, name, change):
    settings=few_recipe(args,change)
    out=root/name
    history=[]
    for step in (250,500,750,1000):
        state=out/'training-state.pt'
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(root/'status.json',{'phase':'training','experiment':name,'target_step':step})
            execute([sys.executable,'-u','-m','distillation.train',*options(settings),
                     '--outdir',out,'--steps',step,*(['--resume',state] if state.exists() else [])],root/f'{name}.log')
        checkpoint=out/'snapshots'/f'student-{step:06d}.ckpt'
        metadata=torch.load(checkpoint,map_location='cpu',weights_only=True)
        torch.testing.assert_close(metadata['state_dict']['backbone.four_sigmas'],torch.tensor([80.,5.,.5,.02]))
        del metadata
        destination=root/'evaluations'/f'{name}-{step}'
        write(root/'status.json',{'phase':'evaluating','experiment':name,'step':step})
        execute(evaluation_command(args,checkpoint,'few_nfe',destination),root/'evaluation.log')
        result=json.loads((destination/'fid.json').read_text())
        row={'step':step,'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),
             'checkpoint_sha256':result['checkpoint_sha256']}
        pin_best(state,row,out)
        history.append(row)
        write(out/'evaluation-history.json',history)
        control=json.loads((args.workdir/f'few-evaluations/{step}/fid.json').read_text())['training_reference_fid']
        if step>=500 and row['fid']>control+1.5:
            write(out/'completion.json',{'phase':'stopped_for_quality',**row})
            return
    write(out/'completion.json',{'phase':'complete',**history[-1]})


def main(args):
    root=args.workdir/'student-followups'
    root.mkdir(parents=True,exist_ok=True)
    write(root/'plan.json',{
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'one_parent_sha256':digest(args.workdir/'baseline/dmd-parent.pt'),
        'few_parent_sha256':digest(args.workdir/'baseline/few_nfe.ckpt'),
        'one_control':'cache16, same original 5k full state, .025 regression, 100 score-only + 650 generator updates',
        'one_arms':{'anchor05':.05,'anchor10':.1},
        'one_unchanged':'16 fitting pairs/class, seed, five fake updates, learning rates, GAN weights, batch',
        'few_control':'few5e6, same exported parent, fresh optimizer, same seed/LR/batch',
        'few_arms':{'few-grid80':{'num_intervals':80},'few-ema995':{'ema_decay':.995}},
        'few_updates_max_per_arm':1000,'few_sampling_grid':[80.,5.,.5,.02],
        'selection':'Legacy training-only monitor; parents may have seen reference images.',
        'official_validation_used':False,'clean_holdout_teacher':'independent, uninterrupted'})
    if args.dry_run:
        print(json.dumps({'one':one_recipe(args,.05),'few':few_recipe(args,{'num_intervals':80})},indent=2))
        return
    # Serialize student jobs beside the independent teacher; never overlap three training jobs.
    upstream=args.workdir/'overnight-one/status.json'
    while True:
        status=json.loads(upstream.read_text())
        if status['phase']=='complete':break
        if status['phase']=='failed':raise RuntimeError(f'Upstream overnight queue failed: {status}')
        write(root/'status.json',{'phase':'waiting_for_overnight','upstream':status})
        time.sleep(30)
    def one_limit(step):
        if step<5500:return None
        p=args.workdir/f'one-followups/evaluations/cache16-{step}/fid.json'
        return json.loads(p.read_text())['training_reference_fid']+2
    # Alternate four- and one-pass arms so both get results early.
    for few_name,change,one_name,weight in (
        ('few-grid80',{'num_intervals':80},'anchor05',.05),
        ('few-ema995',{'ema_decay':.995},'anchor10',.1)):
        if not (root/few_name/'completion.json').exists():
            few_blocks(args,root,few_name,change)
        if not (root/one_name/'completion.json').exists():
            blocks(args,root,one_name,one_recipe(args,weight),args.workdir/'baseline/dmd-parent.pt',
                   (5250,5500,5750),one_limit)
    rows=[{'experiment':p.parent.name,**json.loads(p.read_text())}
          for p in (root/'evaluations').glob('*/fid.json')]
    write(root/'comparison.json',sorted(rows,key=lambda row:(row['nfe'],row['training_reference_fid'])))
    write(root/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=ROOT/'results/nov03')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    args.workdir=args.workdir.resolve()
    try:
        main(args)
    except Exception as error:
        write(args.workdir/'student-followups/status.json',{'phase':'failed','error':str(error)})
        raise
