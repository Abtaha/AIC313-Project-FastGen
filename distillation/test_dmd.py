"""DMD gradient direction/isolation, short resume, and persistent grid tests."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from teacher import EDMTeacher, TeacherConfig
from model import Model
from .model import ConsistencyStudent
from .dmd import distribution_gradient, gradient_surrogate, parser, train, compatible_identity
from .tune_schedule import parser as schedule_parser, run as tune
from .adversarial import GANHead, bottleneck_features, frozen_parameters


class ConstantDenoiser(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, sigma, category):
        return torch.full_like(x, self.value)

    def guided(self, x, sigma, category, guidance):
        return self(x, sigma, category)


class DMDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def teacher(self):
        return EDMTeacher(TeacherConfig(resolution=8, num_classes=3, model_channels=16,
            channel_mult=(1,2), num_blocks=1, attn_resolutions=(4,),dropout=0)).eval().requires_grad_(False)

    def test_gradient_sign_scale_and_detachment(self):
        x = torch.zeros(2,3,8,8,requires_grad=True)
        real, fake = ConstantDenoiser(1), ConstantDenoiser(0)
        gradient = distribution_gradient(x,torch.tensor([0,1]),real,fake,
            sigma=torch.ones(2),noise=torch.zeros_like(x))
        torch.testing.assert_close(gradient,-torch.ones_like(x))
        self.assertFalse(gradient.requires_grad)
        gradient_surrogate(x,gradient).backward()
        # Gradient descent must increase x toward real=1, not toward fake=0.
        self.assertTrue((x.grad<0).all())
        torch.testing.assert_close(x.grad,gradient/x.numel())

    def test_identity_only_allows_ratio_change_when_forking(self):
        legacy = {'teacher_sha256':'a','recipe':{'lr':1e-4}}
        current = {'teacher_sha256':'a','recipe':{'lr':1e-4,'fake_updates':1}}
        changed = {'teacher_sha256':'a','recipe':{'lr':1e-4,'fake_updates':3}}
        self.assertTrue(compatible_identity(legacy,current))
        self.assertFalse(compatible_identity(legacy,changed))
        self.assertTrue(compatible_identity(legacy,changed,fork=True))
        self.assertFalse(compatible_identity(legacy,{'teacher_sha256':'b','recipe':changed['recipe']},fork=True))
        self.assertNotIn('fake_updates',legacy['recipe'])

    def test_gan_features_and_gradient_isolation(self):
        fake = self.teacher()
        fake.requires_grad_(True)
        labels = torch.tensor([0, 2])
        images, sigma = torch.randn(2,3,8,8), torch.tensor([.2, 1.])
        captured = []
        hook = fake.backbone.dec['4x4_in1'].register_forward_hook(lambda m, a, output: captured.append(output))
        fake(images, sigma, labels)
        hook.remove()
        torch.testing.assert_close(bottleneck_features(fake, images, sigma, labels), captured[0], rtol=0, atol=0)
        head = GANHead(fake.config)
        x = images.clamp(-.8,.8).requires_grad_()
        with frozen_parameters(fake, head):
            loss = torch.nn.functional.softplus(-head(fake, x, labels, sigma=sigma, noise=torch.zeros_like(x))).mean()
        loss.backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in list(fake.parameters())+list(head.parameters())))
        detached = x.detach()
        torch.nn.functional.softplus(head(fake, detached, labels)).mean().backward()
        self.assertTrue(any(p.grad is not None for p in fake.parameters()))
        self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_gan_transition_resume_and_training_only_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = self.teacher()
            mapping = {'a':0,'b':1,'c':2}
            base, initial, cache = root/'teacher.ckpt',root/'student.ckpt',root/'cache.pt'
            torch.save({**teacher.checkpoint(),'category_to_id':mapping},base)
            torch.save(ConsistencyStudent.from_teacher(teacher).export(category_to_id=mapping,step=2000,guidance=2),initial)
            labels = torch.tensor([0,0,1,1,2,2])
            torch.save({'identity':{'category_to_id':mapping,'teacher_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
                'guidance':2.},'latents':torch.randn(6,3,8,8),'targets':torch.randn(6,3,8,8),
                'labels':labels,'heldout':torch.tensor([False,True]*3)},cache)
            (root/'train_split.txt').write_text('train-only fixture\n')
            dataset = torch.utils.data.TensorDataset(torch.randn(6,3,8,8).clamp(-1,1),labels)
            dataset.category_ids = labels.tolist()
            args = parser().parse_args(['--teacher',str(base),'--student',str(initial),'--cache',str(cache),
                '--outdir',str(root/'gan'),'--split-dir',str(root),'--steps','1','--batch-size','2','--microbatch','1',
                '--device','cpu','--amp','none','--score-warmup','0','--gan-start','1','--gan-weight','.003'])
            with patch('distillation.adversarial.PokemonDataset', return_value=dataset) as load:
                train(args)
                before=torch.load(root/'gan/training-state.pt',weights_only=True)
                args.resume,args.steps=str(root/'gan/training-state.pt'),2
                train(args)
                self.assertEqual(load.call_args.args[1],root/'train_split.txt')
            after=torch.load(args.resume,weights_only=True)
            self.assertTrue(any(not torch.equal(v,after['gan_head'][k]) for k,v in before['gan_head'].items()))
            self.assertEqual(after['step'],2)
            self.assertTrue(torch.isfinite(torch.tensor([float(p.abs().max()) for p in after['generator'].values()])).all())
            exported=torch.load(root/'gan/one_nfe.ckpt',weights_only=True)
            self.assertFalse(any('gan' in key for key in exported['state_dict']))
            self.assertEqual(exported['training_recipe'],'DMD2-inspired-gan-v1')
            import json
            stats=[json.loads(line) for line in (root/'gan/stats.jsonl').read_text().splitlines()]
            self.assertEqual([row['gan_active'] for row in stats],[False,True])
            self.assertEqual(stats[-1]['reg_weight'],.025)

    def test_grid_persists_and_old_checkpoints_load(self):
        student = ConsistencyStudent.from_teacher(self.teacher()).eval()
        original = student.state_dict()
        original.pop('four_sigmas')
        student.load_state_dict(original)
        student.set_four_sigmas([80,8,.8,.05])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'few.ckpt'
            torch.save(student.export(category_to_id={'a':0,'b':1,'c':2},step=1000,guidance=2),path)
            loaded = Model.load_checkpoint(path,evaluate_mode='few_nfe',config=student.config)
            torch.testing.assert_close(loaded.backbone.four_sigmas,torch.tensor([80,8,.8,.05]))
        with self.assertRaises(ValueError): student.set_four_sigmas([80,1,2,.1])

    def test_training_resume_and_schedule_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teacher = self.teacher()
            mapping = {'a':0,'b':1,'c':2}
            base = root/'teacher.ckpt'
            initial = root/'student.ckpt'
            cache = root/'cache.pt'
            torch.save({**teacher.checkpoint(),'category_to_id':mapping},base)
            student = ConsistencyStudent.from_teacher(teacher).eval()
            torch.save(student.export(category_to_id=mapping,step=1000,guidance=2),initial)
            original = {k:v.clone() for k,v in student.state_dict().items()}
            labels = torch.tensor([0,0,1,1,2,2])
            torch.save({'identity':{'category_to_id':mapping,'teacher_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
                'guidance':2.0},'latents':torch.randn(6,3,8,8),'targets':torch.randn(6,3,8,8).clamp(-1,1),
                'labels':labels,'heldout':torch.tensor([False,True]*3)},cache)
            args = parser().parse_args(['--teacher',str(base),'--student',str(initial),'--cache',str(cache),
                '--outdir',str(root/'dmd'),'--steps','1','--batch-size','2','--microbatch','1',
                '--device','cpu','--amp','none','--score-warmup','0'])
            train(args)
            result = torch.load(root/'dmd'/'one_nfe.ckpt',weights_only=True)
            self.assertTrue(any(not torch.equal(v,result['state_dict']['backbone.'+k])
                                for k,v in original.items() if k!='four_sigmas'))
            args.resume, args.steps = str(root/'dmd'/'training-state.pt'),2
            train(args)
            self.assertEqual(torch.load(args.resume,weights_only=True)['step'],2)
            parent = Path(args.resume)
            parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
            args.resume, args.fork_from, args.outdir = None, str(parent), str(root/'dmd-three')
            args.steps, args.fake_updates = 3, 3
            train(args)
            fork = torch.load(root/'dmd-three'/'training-state.pt',weights_only=True)
            # Parent had two fake updates; the new generator step adds three.
            self.assertEqual(int(next(iter(fork['fake_optimizer']['state'].values()))['step']),5)
            self.assertEqual(int(next(iter(fork['optimizer']['state'].values()))['step']),3)
            self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),parent_hash)
            tune_args = schedule_parser().parse_args(['--student',str(initial),'--cache',str(cache),
                '--outdir',str(root/'schedule'),'--device','cpu','--microbatch','2'])
            tune(tune_args)
            self.assertTrue((root/'schedule'/'few_nfe.ckpt').exists())


if __name__ == '__main__':
    unittest.main()
