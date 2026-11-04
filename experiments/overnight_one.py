"""Bounded overnight one-pass follow-ups with training-only selection and full-state retention."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from types import SimpleNamespace

import torch

from _common import ROOT, options
from experiments.refinement import one_settings, evaluation_command, execute, write, digest


def pin_best(source, row, directory):
    """Pin an atomic full-state inode only when its step matches the evaluated export."""
    record=directory/'best-state.json'
    if record.exists() and json.loads(record.read_text())['fid']<=row['fid']:
        return
    directory.mkdir(parents=True,exist_ok=True)
    temporary=directory/'best-training-state.tmp'
    temporary.unlink(missing_ok=True)
    os.link(source,temporary)
    payload=torch.load(temporary,map_location='cpu',weights_only=True)
    step=payload['step']
    del payload
    if step!=row['step']:
        temporary.unlink()
        return
    os.replace(temporary,directory/'best-training-state.pt')
    write(record,row)


def wait_for_cache16(args, root):
    source=args.workdir/'one-followups'
    initial=source/'best-cache16-state.json'
    if initial.exists():
        pin_best(source/'best-cache16-training-state.pt',json.loads(initial.read_text()),root/'cache16-best')
    while True:
        candidates=[]
        for path in (source/'evaluations').glob('cache16-*/fid.json'):
            row=json.loads(path.read_text())
            candidates.append({'step':int(path.parent.name.rsplit('-',1)[-1]),
                'fid':row['training_reference_fid'],'checkpoint':row['checkpoint'],
                'checkpoint_sha256':row['checkpoint_sha256']})
        if candidates:
            best=min(candidates,key=lambda row:row['fid'])
            pin_best(source/'cache16/training-state.pt',best,root/'cache16-best')
        status=json.loads((source/'cache16-status.json').read_text())
        if status['phase'] in ('complete','stopped_for_quality'):
            return
        if (args.workdir/'one-failed.json').exists():
            raise RuntimeError('Upstream cache16 experiment failed; inspect one-failed.json')
        write(root/'status.json',{'phase':'waiting_for_cache16','upstream':status})
        time.sleep(30)


def blocks(args, root, name, settings, parent, targets, stop_limit):
    out=root/name
    state=out/'training-state.pt'
    history=[]
    for step in targets:
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(root/'status.json',{'phase':'training','experiment':name,'target_step':step})
            restore=['--resume',state] if state.exists() else ['--fork-from',parent]
            execute([sys.executable,'-u','-m','distillation.dmd',*options(settings),
                '--outdir',out,'--steps',str(step),*restore],root/f'{name}.log')
        checkpoint=out/'snapshots'/f'one_nfe-{step:06d}.ckpt'
        destination=root/'evaluations'/f'{name}-{step}'
        write(root/'status.json',{'phase':'evaluating','experiment':name,'step':step})
        execute(evaluation_command(args,checkpoint,'student',destination),root/'evaluation.log')
        result=json.loads((destination/'fid.json').read_text())
        row={'step':step,'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),
             'checkpoint_sha256':result['checkpoint_sha256']}
        pin_best(state,row,out)
        history.append(row)
        write(out/'evaluation-history.json',history)
        if stop_limit(step) is not None and row['fid']>stop_limit(step):
            write(out/'completion.json',{'phase':'stopped_for_quality',**row})
            return False
    write(out/'completion.json',{'phase':'complete',**history[-1]})
    return True


def main(args):
    root=args.workdir/'overnight-one'
    root.mkdir(exist_ok=True)
    write(root/'plan.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_parent_sha256':digest(args.workdir/'baseline/dmd-parent.pt'),
        'teacher_pairs_per_class':64,'teacher_target_count':9815,'teacher_steps':40,'guidance':2.,
        'generator_updates':1000,'score_only_warmup':100,'fake_updates':5,
        'continuation_updates_max_per_arm':500,
        'continuation_gate':'Final checkpoint must be the best, and improve starting FID by at least 1.',
        'selection':'Existing training-only monitoring reference; legacy parents may have seen it.',
        'official_validation_used':False,'fresh_holdout_teacher':'independent run, never interrupted'})
    wait_for_cache16(args,root)
    baseline=json.loads((args.workdir/'student-evaluations/baseline/fid.json').read_text())['training_reference_fid']
    common=one_settings(args)
    # Extend cache16 only if its endpoint is its best and clearly improves the baseline.
    best_file=root/'cache16-best/best-state.json'
    if best_file.exists():
        best=json.loads(best_file.read_text())
        endpoint=args.workdir/'one-followups/evaluations/cache16-6100/fid.json'
        if best['step']==6100 and best['fid']<=baseline-1 and endpoint.exists():
            settings={**common,'cache':str(args.workdir/'one-followups/teacher-cache16.pt'),'fork_refresh_cache':True}
            blocks(args,root,'cache16-extra',settings,root/'cache16-best/best-training-state.pt',
                   (6350,6600),lambda step:best['fid']+.75)
    from teacher import EDMTeacher
    from distillation.support import target_cache
    cache=root/'teacher-cache64.pt'
    write(root/'status.json',{'phase':'generating_teacher_pairs','count':9815})
    torch.cuda.set_per_process_memory_fraction(.30)
    teacher=EDMTeacher.load_checkpoint(common['teacher'],device='cuda')
    mapping=torch.load(common['teacher'],map_location='cpu',weights_only=True)['category_to_id']
    target_cache(SimpleNamespace(cache=str(cache),guidance=2.,teacher_steps=40,seed=123,
        train_per_class=64,microbatch=8,memory_limit_gb=6.),teacher,digest(common['teacher']),mapping,torch.device('cuda'))
    del teacher
    torch.cuda.empty_cache()
    settings={**common,'cache':str(cache),'fork_refresh_cache':True}
    def limit(step):
        if step<5500:return None
        p=args.workdir/f'student-evaluations/fixed5-{step}/fid.json'
        return json.loads(p.read_text())['training_reference_fid']+3
    finished=blocks(args,root,'cache64',settings,args.workdir/'baseline/dmd-parent.pt',
                    (5250,5500,5750,6000,6100),limit)
    best=json.loads((root/'cache64/best-state.json').read_text())
    previous=[json.loads(p.read_text())['training_reference_fid'] for p in (args.workdir/'one-followups/evaluations').glob('*/fid.json')]
    if finished and best['step']==6100 and best['fid']<=min([baseline-1,*previous]):
        blocks(args,root,'cache64-extra',settings,root/'cache64/best-training-state.pt',
               (6350,6600),lambda step:best['fid']+.75)
    rows=[]
    for path in (root/'evaluations').glob('*/fid.json'):
        rows.append({'experiment':path.parent.name,**json.loads(path.read_text())})
    write(root/'comparison.json',sorted(rows,key=lambda row:row['training_reference_fid']))
    write(root/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=ROOT/'results/nov03')
    args=parser.parse_args()
    args.workdir=args.workdir.resolve()
    try:
        main(args)
    except Exception as error:
        write(args.workdir/'overnight-one/status.json',{'phase':'failed','error':str(error)})
        raise
