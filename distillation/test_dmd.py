"""DMD gradient direction/isolation, short resume, and persistent grid tests."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from teacher import EDMTeacher, TeacherConfig
from model import Model
from .model import ConsistencyStudent
from .dmd import distribution_gradient, gradient_surrogate, parser, train
from .tune_schedule import parser as schedule_parser, run as tune


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
            tune_args = schedule_parser().parse_args(['--student',str(initial),'--cache',str(cache),
                '--outdir',str(root/'schedule'),'--device','cpu','--microbatch','2'])
            tune(tune_args)
            self.assertTrue((root/'schedule'/'few_nfe.ckpt').exists())


if __name__ == '__main__':
    unittest.main()
