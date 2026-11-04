"""Reproduce the November teacher-refinement and controlled regression-decay pilots.

All selection references come from train_split.txt. They are excluded from new
GAN updates, but were potentially seen by the pretrained-on-training-split parent
models. These are training-only monitoring scores, not official validation FID.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def prepare(args):
    import os
    import torch
    from dataset import category_from_filename, read_split
    work = args.workdir
    split = ROOT/'data/pokemon-generation-one-22k'
    mapping = json.loads((split/'category_to_id.json').read_text())
    groups = defaultdict(list)
    for path in read_split(split/'train_split.txt'):
        groups[mapping[category_from_filename(path)]].append(path)
    rng = torch.Generator().manual_seed(4317)
    paths = []
    for category in sorted(groups):
        group = groups[category]
        if len(group) <= 10:
            raise ValueError('Need more than 10 training images per class')
        order = torch.randperm(len(group), generator=rng).tolist()
        paths.extend(group[i] for i in order[:10])
    work.mkdir(parents=True, exist_ok=True)
    manifest = work/'monitor.txt'
    text = '\n'.join(paths)+'\n'
    if manifest.exists() and manifest.read_text() != text:
        raise ValueError('Existing monitoring split differs')
    manifest.write_text(text)
    for source, name in [(args.teacher, 'teacher.ckpt'),
                         (args.baseline_root/'checkpoints/dmd-refined-gan/snapshots/one_nfe-005000.ckpt','one_nfe.ckpt'),
                         (args.baseline_root/'checkpoints/consistency-refine-9k-lr2e5-tuned/few_nfe.ckpt','few_nfe.ckpt'),
                         (args.baseline_root/'checkpoints/dmd-refined-gan/training-state.pt','dmd-parent.pt')]:
        target = work/'baseline'/name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            os.link(source, target)
        if digest(target) != digest(source):
            raise ValueError(f'Baseline differs: {name}')
    write(work/'provenance.json', {
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'baseline_hashes':{p.name:digest(p) for p in (work/'baseline').iterdir()},
        'monitor_seed':4317,'reference_count':len(paths),'generated_count':3020,
        'monitor_sha256':digest(manifest), 'monitor_initialization_may_have_seen_images':True,
        'official_validation_used':False,
        'prior_completed_results':{'control':38.65463321095268,'ratio5':39.34358881291132,
                                   'lower_lr':39.28761210108104,'longer_7k':40.19125641230738}})


def execute(command, log):
    print(shlex.join([str(x) for x in command]), flush=True)
    started = time.monotonic()
    with log.open('a') as stream:
        result = subprocess.run([str(x) for x in command],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    write(log.with_suffix('.status.json'), {'returncode':result.returncode,'wall_seconds':time.monotonic()-started})
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def evaluation_command(args, checkpoint, kind, destination, guidance=2.):
    return [sys.executable,'-u','-m','experiments.refinement','--task','evaluate',
            '--baseline-root',args.baseline_root,'--teacher',args.teacher,'--workdir',args.workdir,
            '--checkpoint',checkpoint,'--kind',kind,'--evaluation-output',destination,'--guidance',guidance]


def teacher_run(args):
    from _common import options
    recipe=json.loads((ROOT/'recipes.json').read_text())['teacher']
    recipe.update(lr=2e-5,lr_rampup_kimg=0,microbatch=32,holdout_fraction=.1,
                  loss_every=1000,loss_count=256,fid_every=0,save_every=1000,
                  snapshot_every=1000,state_snapshot_every=5000,guidance=2.,memory_fraction=.52)
    out=args.workdir/'teacher'
    state=out/'training-state.pt'
    import torch
    complete=state.exists() and torch.load(state,map_location='cpu',weights_only=True)['step']>=110000
    if not complete:
        restore=['--resume',state] if state.exists() else ['--init-teacher',args.workdir/'baseline/teacher.ckpt']
        execute([sys.executable,'-u','-m','training.teacher',*options(recipe),'--outdir',out,
                 '--steps','110000',*restore],args.workdir/'teacher.log')
    for checkpoint in [args.workdir/'baseline/teacher.ckpt',
                       *[out/'snapshots'/f'teacher-{step:07d}.ckpt' for step in (101000,105000,110000)]]:
        for guidance in (1.5,2.,2.5):
            destination=args.workdir/'teacher-evaluations'/f'{checkpoint.stem}-g{guidance}'
            execute(evaluation_command(args,checkpoint,'teacher',destination,guidance),args.workdir/'teacher-evaluation.log')
    write(args.workdir/'teacher-complete.json',{'phase':'complete'})


def student_run(args):
    from _common import options
    parent=json.loads((args.baseline_root/'checkpoints/dmd-refined-gan/config.json').read_text())['arguments']
    keys=('lr','fake_lr','guidance','reg_weight','score_warmup','gan_weight','gan_start','gan_reg_weight','amp','seed','batch_size')
    recipe={k:parent[k] for k in keys}
    for key in ('student','cache'):
        recipe[key]=str((args.baseline_root/parent[key]).resolve())
    recipe.update(teacher=str(args.workdir/'baseline/teacher.ckpt'),microbatch=8,fake_updates=5,
                  memory_fraction=.34,memory_limit_gb=7.,log_every=25,save_every=250,
                  generator_freeze_until=5100,diagnostic_every=250,diagnostic_count=32,
                  gan_exclude_manifest=str(args.workdir/'monitor.txt'),
                  data_root=str(ROOT/'data/pokemon-generation-one-22k'),split_dir=str(ROOT/'data/pokemon-generation-one-22k'))
    execute(evaluation_command(args,args.workdir/'baseline/one_nfe.ckpt','student',args.workdir/'student-evaluations/baseline'),args.workdir/'student-evaluation.log')
    # Same full parent state, critic warmup, seed, batch, and 1,000 generator updates.
    for name, decay in [('decay5',True),('fixed5',False)]:
        out=args.workdir/name
        state=out/'training-state.pt'
        import torch
        complete=state.exists() and torch.load(state,map_location='cpu',weights_only=True)['step']>=6100
        if not complete:
            restore=['--resume',state] if state.exists() else ['--fork-from',args.workdir/'baseline/dmd-parent.pt']
            settings={**recipe,'reg_decay_start':5100 if decay else 0,'reg_decay_end':5600 if decay else 0}
            execute([sys.executable,'-u','-m','distillation.dmd',*options(settings),'--outdir',out,
                     '--steps','6100',*restore],args.workdir/f'{name}.log')
        for step in (5250,5500,5750,6000,6100):
            checkpoint=out/'snapshots'/f'one_nfe-{step:06d}.ckpt'
            execute(evaluation_command(args,checkpoint,'student',args.workdir/f'student-evaluations/{name}-{step}'),args.workdir/'student-evaluation.log')
    rows=[]
    for path in (args.workdir/'student-evaluations').glob('*/fid.json'):
        rows.append({'experiment':path.parent.name,**json.loads(path.read_text())})
    write(args.workdir/'student-comparison.json',sorted(rows,key=lambda row:row['training_reference_fid']))
    write(args.workdir/'student-complete.json',{'phase':'complete'})


def few_run(args):
    from _common import options
    import torch
    recipes=json.loads((ROOT/'recipes.json').read_text())
    settings={**recipes['consistency'],**recipes['refinement']}
    settings.update(lr=5e-6,microbatch=16,save_every=250,sample_every=250,log_every=25,
                    teacher=str(args.workdir/'baseline/teacher.ckpt'),
                    init_student=str(args.workdir/'baseline/few_nfe.ckpt'),
                    data_root=str(ROOT/'data/pokemon-generation-one-22k'),
                    split_dir=str(ROOT/'data/pokemon-generation-one-22k'))
    out=args.workdir/'few5e6'
    state=out/'training-state.pt'
    complete=state.exists() and torch.load(state,map_location='cpu',weights_only=True)['step']>=2000
    if not complete:
        restore=['--resume',state] if state.exists() else []
        execute([sys.executable,'-u','-m','distillation.train',*options(settings),
                 '--outdir',out,'--steps','2000',*restore],args.workdir/'few5e6.log')
    for name,checkpoint in [('baseline',args.workdir/'baseline/few_nfe.ckpt'),
            *[(str(step),out/'snapshots'/f'student-{step:06d}.ckpt') for step in range(250,2001,250)]]:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        grid=payload['state_dict']['backbone.four_sigmas']
        torch.testing.assert_close(grid,torch.tensor([80.,5.,.5,.02]))
        destination=args.workdir/'few-evaluations'/name
        execute(evaluation_command(args,checkpoint,'few_nfe',destination),args.workdir/'few-evaluation.log')
    rows=[]
    for path in (args.workdir/'few-evaluations').glob('*/fid.json'):
        rows.append({'experiment':path.parent.name,**json.loads(path.read_text())})
    write(args.workdir/'few-comparison.json',sorted(rows,key=lambda row:row['training_reference_fid']))
    write(args.workdir/'few-complete.json',{'phase':'complete'})


def one_run(args):
    """Reuse the completed fixed5 control; change one objective/data factor per arm."""
    from _common import options
    import torch
    parent=json.loads((args.baseline_root/'checkpoints/dmd-refined-gan/config.json').read_text())['arguments']
    keys=('lr','fake_lr','guidance','reg_weight','score_warmup','gan_weight','gan_start','gan_reg_weight','amp','seed','batch_size')
    common={k:parent[k] for k in keys}
    for key in ('student','cache'):
        common[key]=str((args.baseline_root/parent[key]).resolve())
    common.update(teacher=str(args.workdir/'baseline/teacher.ckpt'),microbatch=8,fake_updates=5,
                  memory_fraction=.34,memory_limit_gb=7.,log_every=25,save_every=250,
                  generator_freeze_until=5100,diagnostic_every=250,diagnostic_count=32,
                  gan_exclude_manifest=str(args.workdir/'monitor.txt'),
                  data_root=str(ROOT/'data/pokemon-generation-one-22k'),split_dir=str(ROOT/'data/pokemon-generation-one-22k'))
    output=args.workdir/'one-followups'
    output.mkdir(exist_ok=True)
    write(output/'plan.json',{'parent_sha256':digest(args.workdir/'baseline/dmd-parent.pt'),
        'control':'fixed5, same parent, 100 score-only warmup + 1000 generator updates',
        'arms':{'gan-g01':{'generator_gan_weight':.01,'critic_gan_weight':.003},
                'cache16':{'fitting_pairs_per_class':16,'original_pairs_per_class':2}},
        'early_stop':'After 400 generator updates, stop an arm if monitoring FID exceeds its matched-step control by more than 3.',
        'official_validation_used':False})
    for name in ('gan-g01','cache16'):
        status=output/f'{name}-status.json'
        if status.exists() and json.loads(status.read_text()).get('phase') in ('complete','stopped_for_quality'):
            continue
        settings=dict(common)
        if name=='gan-g01':
            settings['generator_gan_weight']=.01
        else:
            from types import SimpleNamespace
            from teacher import EDMTeacher
            from distillation.support import target_cache
            path=output/'teacher-cache16.pt'
            cache_args=SimpleNamespace(cache=str(path),guidance=2.,teacher_steps=40,seed=123,
                                       train_per_class=16,microbatch=8,memory_limit_gb=6.)
            write(status,{'phase':'generating_teacher_pairs','count':2567})
            torch.cuda.set_per_process_memory_fraction(.30)
            teacher=EDMTeacher.load_checkpoint(settings['teacher'],device='cuda')
            metadata=torch.load(settings['teacher'],map_location='cpu',weights_only=True)
            target_cache(cache_args,teacher,digest(settings['teacher']),metadata['category_to_id'],torch.device('cuda'))
            del teacher,metadata
            torch.cuda.empty_cache()
            settings.update(cache=str(path),fork_refresh_cache=True)
        out=output/name
        state=out/'training-state.pt'
        stopped=False
        for step in (5250,5500,5750,6000,6100):
            saved=torch.load(state,map_location='cpu',weights_only=True)['step'] if state.exists() else 5000
            if saved<step:
                write(status,{'phase':'training','target_step':step,'starting_step':saved})
                restore=['--resume',state] if state.exists() else ['--fork-from',args.workdir/'baseline/dmd-parent.pt']
                execute([sys.executable,'-u','-m','distillation.dmd',*options(settings),'--outdir',out,
                         '--steps',str(step),*restore],output/f'{name}.log')
            checkpoint=out/'snapshots'/f'one_nfe-{step:06d}.ckpt'
            destination=output/'evaluations'/f'{name}-{step}'
            write(status,{'phase':'evaluating','step':step})
            execute(evaluation_command(args,checkpoint,'student',destination),output/'evaluation.log')
            result=json.loads((destination/'fid.json').read_text())
            control=json.loads((args.workdir/f'student-evaluations/fixed5-{step}/fid.json').read_text())
            if step>=5500 and result['training_reference_fid']>control['training_reference_fid']+3:
                write(status,{'phase':'stopped_for_quality','step':step,
                    'training_reference_fid':result['training_reference_fid'],
                    'matched_control_fid':control['training_reference_fid']})
                stopped=True
                break
        if not stopped:
            write(status,{'phase':'complete','step':6100})
    rows=[]
    for path in (output/'evaluations').glob('*/fid.json'):
        rows.append({'experiment':path.parent.name,**json.loads(path.read_text())})
    write(output/'comparison.json',sorted(rows,key=lambda row:row['training_reference_fid']))
    write(args.workdir/'one-complete.json',{'phase':'complete'})


def evaluate(args):
    import tempfile
    import torch
    from PIL import Image
    from dataset import PokemonDataset, read_split
    from evaluate import compute_fid, generate_samples
    from torchvision.transforms.functional import to_pil_image, pil_to_tensor
    from torchvision.utils import save_image
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.34)
    destination=args.evaluation_output
    destination.mkdir(parents=True,exist_ok=True)
    result=destination/'fid.json'
    checkpoint_hash=digest(args.checkpoint)
    if result.exists():
        if json.loads(result.read_text())['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('Evaluation checkpoint changed')
        return
    split=ROOT/'data/pokemon-generation-one-22k'
    mapping=json.loads((split/'category_to_id.json').read_text())
    if args.kind=='teacher':
        from teacher import EDMTeacher
        teacher=EDMTeacher.load_checkpoint(args.checkpoint,device='cuda')
        class Adapter:
            def eval(self):
                return self
            def sample(self, shape, **kwargs):
                return teacher.sample(shape,num_steps=40,guidance=args.guidance,**kwargs)
        model=Adapter()
    else:
        from model import Model
        model=Model.load_checkpoint(args.checkpoint,evaluate_mode='few_nfe' if args.kind=='few_nfe' else 'one_nfe',device='cuda')
    reference=args.workdir/'reference'
    # Each evaluator has its own reference directory to avoid partial shared exports.
    reference=destination/'reference'
    paths=read_split(args.workdir/'monitor.txt')
    dataset=PokemonDataset(ROOT/'data/pokemon-generation-one-22k',args.workdir/'monitor.txt',category_to_id=mapping)
    reference.mkdir(exist_ok=True)
    for index in range(len(dataset)):
        to_pil_image(((dataset[index]+1)/2).clamp(0,1)).convert('RGB').save(reference/f'{index:04d}.png')
    categories=torch.arange(len(mapping)).repeat_interleave(20)
    torch.manual_seed(10789)
    with tempfile.TemporaryDirectory(prefix='generated-',dir=destination) as directory:
        generated=generate_samples(model,Path(directory),categories,batch_size=8,device='cuda')
        score=compute_fid(generated,reference,device='cuda',batch_size=8)
        names=['Abra','Charizard','Eevee','Gengar','Mewtwo','Pikachu','Snorlax','Squirtle']
        ids=[mapping[name] for name in names if name in mapping]
        if not ids:
            ids=list(range(0,len(mapping),max(len(mapping)//8,1)))[:8]
        tiles=[]
        for category in ids:
            for index in range(8):
                with Image.open(generated/f'{category:04d}_{index:02d}.png') as image:
                    tiles.append(pil_to_tensor(image.convert('RGB')).float()/255)
        save_image(torch.stack(tiles),destination/'samples.png',nrow=8)
        write(destination/'sample-rows.json',{'classes':[name for name,index in mapping.items() if index in ids]})
    for path in reference.glob('*.png'):
        path.unlink()
    reference.rmdir()
    row={'training_reference_fid':score,'reference_count':len(paths),'generated_count':len(categories),
         'seed':10789,'batch_size':8,'checkpoint':str(args.checkpoint),'checkpoint_sha256':checkpoint_hash,
         'kind':args.kind,'nfe':158 if args.kind=='teacher' else (4 if args.kind=='few_nfe' else 1),
         'guidance':args.guidance if args.kind=='teacher' else 1.,
         'monitor_sha256':digest(args.workdir/'monitor.txt'),'official_validation_used':False,
         'note':'Training-only reference; parent models may have seen these images. Not official FID.'}
    write(result,row)
    print(json.dumps(row),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['prepare','teacher','student','few','one','evaluate'],required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--kind',choices=['teacher','student','few_nfe'])
    parser.add_argument('--evaluation-output',type=Path)
    parser.add_argument('--guidance',type=float,default=2.)
    args=parser.parse_args()
    args.workdir=args.workdir.resolve()
    if args.task=='prepare': prepare(args)
    elif args.task=='evaluate': evaluate(args)
    else:
        (args.workdir/f'{args.task}-failed.json').unlink(missing_ok=True)
        try:
            {'teacher':teacher_run,'student':student_run,'few':few_run,'one':one_run}[args.task](args)
        except Exception as error:
            write(args.workdir/f'{args.task}-failed.json',{'error':str(error)})
            raise


if __name__=='__main__': main()
