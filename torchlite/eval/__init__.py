from torchlite.data.datasets.srpgan import EvalDataset
from torchlite.torch.models.srpgan import Generator
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.train_callbacks import ModelSaverCallback
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def srpgan_eval(images, generator_file, upscale_factor, use_cuda, num_workers=os.cpu_count()):
    """
    Turn a list of images to super resolution and returns them
    Args:
        num_workers (int): Number of processors to use
        use_cuda (bool): Whether or not to use the GPU
        upscale_factor (int): Either 2, 4 or 8
        images (list): List of Pillow images
        generator_file (file): The generator saved model file

    Returns:
        list: A list of SR images
    """
    netG = Generator(upscale_factor)
    learner = Learner(ClassifierCore(netG, None, None), use_cuda=use_cuda)
    ModelSaverCallback.restore_model_from_file(netG, generator_file, load_with_cpu=not use_cuda)
    eval_ds = EvalDataset(images)
    # One batch at a time as the pictures may differ in size
    eval_dl = DataLoader(eval_ds, 1, shuffle=False, num_workers=num_workers)

    images_pred = []
    predictions = learner.predict(eval_dl, flatten_predictions=False)
    tfs = transforms.Compose([
        transforms.ToPILImage(),
    ])
    for pred in predictions:
        pred = pred.view(pred.size()[1:])  # Remove batch size == 1
        images_pred.append(tfs(pred.cpu()))

    return images_pred
