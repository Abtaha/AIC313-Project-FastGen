"""Four-pass backward simulation with DMD/GAN: bounded pure and anchored pilots."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from _common import ROOT, options
from experiments.refinement import digest, execute, evaluation_command, write, one_settings
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


def one_conditioning(args):
    from distillation.model import ConsistencyStudent
    source=args.workdir/'multistep-dmd/anchored/snapshots/few_nfe-000350.ckpt'
    root=args.workdir/'one-conditioning';root.mkdir(parents=True,exist_ok=True)
    write(root/'plan.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':digest(source),'source_four_pass_validation_fid':21.7534219894452,
        'hypothesis':'Lower fixed conditioning may provide a better one-pass generator initialization.',
        'levels':[80.,5.,1.],'selection':'Training-only reference, same seed/batch. No official-reference sweep.',
        'nfe':1,'generator_updates_max':1000,'critic_warmup':100,'regression':.025,
        'reference':'DMD2 F.4 motivates testing conditioning shifts; this EDM adaptation is experimental.'})
    if args.dry_run:return
    probes=root/'probes';probes.mkdir(exist_ok=True)
    rows=[]
    for level in (80.,5.,1.):
        checkpoint=probes/f'sigma-{level:g}.ckpt'
        if not checkpoint.exists():
            student=ConsistencyStudent.load_checkpoint(source)
            metadata=torch.load(source,map_location='cpu',weights_only=True)
            student.set_one_sigma(level)
            payload=student.export(category_to_id=metadata['category_to_id'],step=0,guidance=2.)
            payload['conditioning_source_sha256']=digest(source)
            torch.save(payload,checkpoint)
            del student,metadata,payload
        destination=root/'evaluations'/f'probe-{level:g}'
        write(root/'status.json',{'phase':'probing','sigma':level})
        execute(evaluation_command(args,checkpoint,'student',destination),root/'evaluation.log')
        result=json.loads((destination/'fid.json').read_text())
        rows.append({'sigma':level,'fid':result['training_reference_fid'],'step':0,
                     'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint)})
    parent=min(rows,key=lambda row:row['fid']);write(root/'probe-results.json',rows)
    baseline=json.loads((args.workdir/'student-followups/evaluations/anchor05-5500/fid.json').read_text())
    if parent['fid']>=baseline['training_reference_fid']:
        write(root/'quality-gate.json',{'decision':'Reject weaker frozen initializations; use protected best one-pass full state.',
            'best_probe_internal_fid':parent['fid'],'baseline_internal_fid':baseline['training_reference_fid']})
        score_noise(args,root,baseline)
        return
    recipe={**settings(args,.025),'student':parent['checkpoint'],'sampling_nfe':1,
            'lr':5e-6,'save_every':1000000,'seed':789}
    out=root/'train';state=out/'training-state.pt';history=[parent]
    for step in (100,350,600,850,1100):
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(root/'status.json',{'phase':'training','sigma':parent['sigma'],'target_step':step})
            execute([sys.executable,'-u','-m','distillation.dmd',*options(recipe),'--outdir',out,
                     '--steps',step,*(['--resume',state] if state.exists() else [])],root/'train.log')
        if step==100:continue
        checkpoint=out/'snapshots'/f'one_nfe-{step:06d}.ckpt'
        destination=root/'evaluations'/f'train-{step}'
        execute(evaluation_command(args,checkpoint,'student',destination),root/'evaluation.log')
        result=json.loads((destination/'fid.json').read_text())
        row={'step':step,'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint)}
        previous=min(history,key=lambda row:row['fid'])
        if row['fid']<previous['fid']:write(out/'best-state.json',row)
        history.append(row)
        write(root/'history.json',history)
        if row['fid']>previous['fid']+2:break
    selected=min(history,key=lambda row:row['fid']);write(root/'selected.json',selected)
    official(args,root,Path(selected['checkpoint']),'one_nfe','selected-one')
    write(root/'status.json',{'phase':'complete'})


def score_noise(args,root,baseline):
    parent=args.workdir/'student-followups/anchor05/best-training-state.pt'
    original=json.loads((parent.parent/'config.json').read_text())['arguments']
    keys=('teacher','student','cache','data_root','split_dir','lr','fake_lr','guidance','reg_weight',
          'score_warmup','gan_weight','gan_start','amp','seed','batch_size','fake_updates',
          'generator_freeze_until','gan_exclude_manifest')
    recipe={**{k:original[k] for k in keys},'microbatch':8,'memory_fraction':.34,
            'memory_limit_gb':7.,'log_every':25,'save_every':1000000,'diagnostic_every':250,'diagnostic_count':16,
            'fork_refresh_cache':True,'gan_reg_weight':.05,'dm_p_mean':0.,'dm_p_std':1.2}
    out=root/'score-noise';state=out/'training-state.pt'
    initial={'step':5500,'fid':baseline['training_reference_fid'],'checkpoint':baseline['checkpoint'],
             'checkpoint_sha256':baseline['checkpoint_sha256']}
    history=[initial]
    write(root/'score-noise-plan.json',{'parent_sha256':digest(parent),'changed_factor':'Generator score corruption log-sigma mean -1.2 -> 0.0; std 1.2 unchanged.',
        'hypothesis':'More moderate/high-noise score supervision may help global structure; unproven adaptation.',
        'fake_training_distribution':'Unchanged original EDM loss.','recipe':recipe,
        'max_generator_updates':1000,'extension_gate':'After 500, continue only if internal FID improved by at least .5.',
        'selection':'Training-only reference; original teacher and legacy parents may have seen it.'})
    for step in (5750,6000,6250,6500):
        saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 0
        if saved<step:
            write(root/'status.json',{'phase':'score_noise_training','target_step':step})
            restore=['--resume',state] if state.exists() else ['--fork-from',parent]
            execute([sys.executable,'-u','-m','distillation.dmd',*options(recipe),
                     '--outdir',out,'--steps',step,*restore],root/'score-noise.log')
        checkpoint=out/'snapshots'/f'one_nfe-{step:06d}.ckpt'
        destination=root/'evaluations'/f'score-noise-{step}'
        execute(evaluation_command(args,checkpoint,'student',destination),root/'evaluation.log')
        result=json.loads((destination/'fid.json').read_text())
        row={'step':step,'fid':result['training_reference_fid'],'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint)}
        history.append(row);write(root/'score-noise-history.json',history)
        if row['fid']>initial['fid']+.75:break
        if step>=6000 and min(r['fid'] for r in history)>initial['fid']-.5:break
    selected=min(history,key=lambda row:row['fid']);write(root/'selected.json',selected)
    if selected['checkpoint']!=initial['checkpoint']:
        official(args,root,Path(selected['checkpoint']),'one_nfe','selected-score-noise')
    write(root/'status.json',{'phase':'complete'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,default=ROOT/'results/nov03')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--task',choices=('four','one-conditioning'),default='four')
    args=parser.parse_args();args.workdir=args.workdir.resolve()
    try:(one_conditioning if args.task=='one-conditioning' else main)(args)
    except Exception as error:
        name='one-conditioning' if args.task=='one-conditioning' else 'multistep-dmd'
        write(args.workdir/name/'status.json',{'phase':'failed','error':str(error)})
        raise
