import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

from TwoOutput3DResNet.Transformations.temporal_transforms import TemporalIndexCrop
from TwoOutput3DResNet.datasets.singleEvaluation import SingleEvaluation

SAMPLE_DURATION = 16
ROOT_PATH = '/media/datasets/Video/UCF_CRIME/jpg/'


def neural_predicate(network, folder, start_frame, spatial_transform=None):
    # Step 1: Get the input data for the network (knowing that it has to be id i)
    # Step 2: Make the required transformations to the input to be able to feed it to the network
    # Step 3: Feed the input to network.net and get the output
    # Step 4: Make the required transformations to the output and return it as the probabilities
    data = SingleEvaluation(
        root_path=os.path.join(ROOT_PATH, str(folder)[1:].split('.')[0]),
        annotation_path=None,
        subset=None,
        n_samples_for_each_video=1,
        spatial_transform=spatial_transform,
        temporal_transform=TemporalIndexCrop(SAMPLE_DURATION, int(start_frame)),
        target_transform=None,
        sample_duration=SAMPLE_DURATION,
        get_loader=None
    )

    loader = torch.utils.data.DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    input_frames, target = next(iter(loader))

    network_input = Variable(input_frames)
    output = network.net(network_input)
    # output = torch.tensor([[0, 1]])

    return output.squeeze(0)
