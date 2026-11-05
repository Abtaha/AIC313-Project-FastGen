"""Bounded one-pass feature-anchor/control pilots and final official reports."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _common import ROOT
from experiments.overnight_one import blocks, pin_best
from experiments.refinement import digest, execute, one_settings, write
from experiments.student_followups import few_blocks


def wait_for(path, status_path):
    while True:
        row=json.loads(path.read_text())
        if row['phase']=='complete':return
        if row['phase']=='failed':raise RuntimeError(f'Upstream failed: {path}: {row}')
        write(status_path,{'phase':'waiting','upstream':str(path),'status':row})
        time.sleep(30)


def official(args, root, checkpoint, kind, name):
    destination=root/'official'/name
    protocol=root/'official'/f'{name}-protocol.json'
    seed,batch=(10789,8) if kind=='one_nfe' else (0,32)
    if kind=='teacher':batch=8
    identity={'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),
              'kind':kind,'seed':seed,'batch_size':batch,'reference_count':3020,
              'selection':'Training-only monitoring selected student; fixed endpoint teacher.',
              'purpose':'Reporting only; no adaptive validation sweep.'}
    if (destination/'fid.json').exists():
        if json.loads(protocol.read_text())!=identity:raise ValueError('Report identity changed')
        return
    write(protocol,identity)
    write(root/'status.json',{'phase':'official_reporting','model':name})
    execute([sys.executable,'-u','evaluate_models.py','--kind',kind,'--checkpoint',checkpoint,
             '--outdir',destination,'--seed',seed,'--batch-size',batch],root/f'{name}-official.log')


def main(args):
    root=args.workdir/'overnight-finish'
    root.mkdir(parents=True,exist_ok=True)
    parent=args.workdir/'student-followups/anchor05'
    best=json.loads((parent/'best-state.json').read_text())
    source=parent/'best-training-state.pt'
    if best['step']!=5500:raise ValueError('Expected protected anchor05 step-5500 parent')
    write(root/'plan.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'parent':best,'full_parent_sha256':digest(source),'arms':{'control':0.,'teacher-features':.02},
        'objective':'Same DMD/GAN + .05 pixel regression; experimental .02 spatial teacher-feature cosine anchor in treatment only.',
        'teacher_features':'Frozen original teacher encoder/middle at sigma .5, identical pair corruption; no external pretrained model.',
        'generator_updates_max_per_arm':1000,'checks':[5750,6000,6250,6500],
        'stop':'FID > parent+.75 initially, > parent+.25 from 500 extra updates.',
        'official_validation_used_for_selection':False,'selection':'Legacy training-only monitor; parents may have seen it.',
        'four_extra_gate':'At most 500 more updates if grid80 step-2000 is its best and improves internal parent FID by .15.',
        'official_reports':'Internally selected one-pass, internally selected four-pass, fixed final clean teacher.'})
    if args.dry_run:return
    wait_for(args.workdir/'trajectory-four/status.json',root/'status.json')
    extension=args.workdir/'four-extension/grid80'
    four_best=json.loads((extension/'best-state.json').read_text())
    old_four=json.loads((args.workdir/'student-followups/few-grid80/best-state.json').read_text())
    if four_best['step']==2000 and four_best['fid']<=old_four['fid']-.15:
        out=root/'four-extra';out.mkdir(exist_ok=True)
        if not (out/'training-state.pt').exists():
            os.link(extension/'best-training-state.pt',out/'training-state.pt')
        pin_best(extension/'best-training-state.pt',four_best,out)
        if not (out/'completion.json').exists():
            few_blocks(args,root,'four-extra',{'num_intervals':80},targets=(2250,2500),fixed_limit=four_best['fid']+.5)
    settings={**one_settings(args),'cache':str(args.workdir/'one-followups/teacher-cache16.pt'),
              'fork_refresh_cache':True,'gan_reg_weight':.05}
    for name,weight in [('control',0.),('teacher-features',.02)]:
        if (root/name/'completion.json').exists():continue
        pin_best(source,best,root/name)
        blocks(args,root,name,{**settings,'teacher_feature_weight':weight},source,
               (5750,6000,6250,6500),lambda step:best['fid']+(.75 if step<6000 else .25))
    choices=[best,*[json.loads(p.read_text()) for p in root.glob('*/best-state.json')]]
    one=min(choices,key=lambda row:row['fid'])
    four_paths=[args.workdir/'student-followups/few-grid80/best-state.json',
                args.workdir/'four-extension/grid80/best-state.json',
                root/'four-extra/best-state.json',
                *list((args.workdir/'trajectory-four').glob('*/best-state.json'))]
    four=min([json.loads(p.read_text()) for p in four_paths if p.exists()],key=lambda row:row['fid'])
    write(root/'selected.json',{'one':one,'four':four,'selection':'Internal FID only, before official reporting.'})
    official(args,root,Path(one['checkpoint']),'one_nfe','selected-one')
    official(args,root,Path(four['checkpoint']),'few_nfe','selected-four')
    # Finish the teacher as well; never sample a checkpoint that can change mid-evaluation.
    teacher_run=ROOT/'runs/holdout-v1/teacher'
    while not (teacher_run/'snapshots/teacher-0100000.ckpt').exists():
        if (ROOT/'results/holdout-v1/exit-status').exists():
            raise RuntimeError('Teacher exited without its final step-100000 snapshot')
        write(root/'status.json',{'phase':'waiting_for_teacher_100k'})
        time.sleep(30)
    official(args,root,teacher_run/'snapshots/teacher-0100000.ckpt','teacher','teacher-100k')
    reports=[{'experiment':p.parent.name,**json.loads(p.read_text())} for p in (root/'official').glob('*/fid.json')]
    write(root/'official-comparison.json',reports)
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
        write(args.workdir/'overnight-finish/status.json',{'phase':'failed','error':str(error)})
        raise
