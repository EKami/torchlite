from torchlite.nn.models.srgan import Generator
from torchlite.nn.learners.learner import Learner
from torchlite.nn.learners.cores import ClassifierCore
from torchlite.nn.train_callbacks import ModelSaverCallback
from torchlite.data.datasets.srgan import EvalDataset
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def srgan_eval(images, models_dir, upscale_factor, use_cuda, num_workers=os.cpu_count()):
    """
    Turn a list of images to super resolution and returns them
    Args:
        num_workers (int): Number of processors to use
        use_cuda (bool): Whether or not to use the GPU
        upscale_factor (int): Either 2, 4 or 8
        images (list): List of Pillow images
        models_dir (str): Path to the saved models

    Returns:
        list: A list of SR images
    """
    netG = Generator(upscale_factor)
    learner = Learner(ClassifierCore(netG, None, None), use_cuda=use_cuda)
    ModelSaverCallback.restore_model([netG], models_dir, load_with_cpu=not use_cuda)
    eval_ds = EvalDataset(images)
    # One batch at a time as the pictures may differ in size
    eval_dl = DataLoader(eval_ds, 1, shuffle=False, num_workers=num_workers)

    images_pred = []
    predictions = learner.predict(eval_dl)
    for i, pred in enumerate(predictions):
        pred = pred.view(pred.size()[1:])  # Remove batch size == 1
        images_pred.append(transforms.ToPILImage()(pred.cpu()))

    return images_pred