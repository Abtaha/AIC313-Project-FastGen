"""Train a fresh V1 teacher with a fixed class-balanced, training-only holdout."""
from _common import ROOT, RECIPES, options, parser, paths, stage
from training.holdout import partition


def main():
    command=parser(__doc__,teacher=False)
    command.set_defaults(split_dir=ROOT/'data/holdout-v1',workdir=ROOT/'runs/holdout-v1')
    command.add_argument('--steps',type=int,default=100000)
    args=command.parse_args()
    data=paths(args)
    split=partition(args.split_dir)
    if split is None:
        raise ValueError('A validated clean holdout partition is required')
    print(f"Fresh teacher: {split['fit_count']} fitting images, {split['holdout_count']} untouched monitoring images.",flush=True)
    recipe={**RECIPES['teacher'],'microbatch':32,'memory_fraction':.52,
            'holdout_fraction':0.,'loss_every':2000,'loss_count':512,
            'fid_every':10000,'fid_count':1510,'fid_steps':18,'guidance':2.,
            'snapshot_every':5000,'state_snapshot_every':20000}
    stage('training.teacher',[*data,*options(recipe)],args.workdir/'teacher',args.steps,
          dry_run=args.dry_run,resume=args.resume)


if __name__=='__main__':main()
