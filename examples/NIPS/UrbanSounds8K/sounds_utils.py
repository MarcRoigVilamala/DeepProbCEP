import os

import librosa
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

try:
    import net as net_module
    import data as data_module
    from utils import load_audio
except ImportError:
    pass


def _get_transform(config, name):
    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    return getattr(data_module, tsf_name)(name, tsf_args)


class SoundVGGish(nn.Module):
    def __init__(self, n_classes=10):
        super(SoundVGGish, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=128, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=n_classes),

            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def neural_predicate_vggish(network, path, use_saved_preprocessed=True):
    preprocessed_filename = 'vggish_preprocessed/' + os.path.splitext(os.path.basename(str(path)[1:-1]))[0] + '.pt'

    if use_saved_preprocessed and os.path.exists(preprocessed_filename):
        # Try to load the features from a file
        features = torch.load(preprocessed_filename)
    else:
        print(path, preprocessed_filename)

        raise Exception

        vggish_net = torch.hub.load('harritaylor/torchvggish', 'vggish')
        vggish_net.eval()

        features = vggish_net(path)

        if len(features.shape) == 2:
            features = features[0]

        if use_saved_preprocessed:
            torch.save(features, preprocessed_filename)

    features = features.unsqueeze(0)

    # Use the neural network to get the classification of the audio
    res = network.net(features.float())

    # Return the result
    return res.squeeze(0)


class SoundLinearNet(nn.Module):
    def __init__(self, input_size=200, n_classes=10):
        super(SoundLinearNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=256, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


# class SoundCNNet(nn.Module):
#     def __init__(self, n_classes=10):
#         super(SoundCNNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(40, 64, kernel_size=5, padding=2, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(5, padding=2),
#
#             nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(5, padding=2),
#             nn.Dropout(0.3),
#
#             nn.Flatten(),
#
#             nn.Linear(in_features=128, out_features=256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#
#             nn.Linear(in_features=256, out_features=512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#
#             nn.Linear(in_features=512, out_features=n_classes),
#
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         return self.net(x.view(-1, 40, 5, 1))
class SoundCNNet(nn.Module):
    def __init__(self, n_classes=10):
        super(SoundCNNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=(0, 1)),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=(0, 1)),
            nn.Dropout(0.3),

            nn.Flatten(),

            nn.Linear(in_features=2560, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(in_features=512, out_features=n_classes),

            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x.view(-1, 40, 5, 1).permute(0, 3, 1, 2))


def neural_predicate_linear(network, path, use_saved_preprocessed=True):
    preprocessed_filename = 'preprocessed/' + os.path.splitext(os.path.basename(str(path)[1:-1]))[0] + '.pt'

    if use_saved_preprocessed and os.path.exists(preprocessed_filename):
        # Try to load the features from a file
        features = torch.load(preprocessed_filename)
    else:
        # Load the file
        y, sr = librosa.load(str(path)[1:-1])

        # Process the file to get the features
        mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)

        # Combine the features
        features = torch.from_numpy(
            np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), 200)
        )

        features = features.unsqueeze(0)

        if use_saved_preprocessed:
            torch.save(features, preprocessed_filename)

    # Use the neural network to get the classification of the audio
    res = network.net(features.float())

    # Return the result
    return res.squeeze(0)


class SoundsUtils(object):
    def __init__(self, config):
        self.config = config

        self.t_transforms = _get_transform(self.config, 'train')
        self.v_transforms = _get_transform(self.config, 'val')

        print(self.v_transforms)

        data_manager = getattr(data_module, self.config['data']['type'])(self.config['data'])
        classes = np.array(['air_conditioner', 'car_horn'])

        # self.t_loader = data_manager.get_loader('train', self.t_transforms)
        # self.v_loader = data_manager.get_loader('val', self.v_transforms)

        m_name = self.config['model']['type']
        self.network = getattr(net_module, m_name)(classes, config=self.config)
        num_classes = len(classes)

        print(self.network)

        # loss = getattr(net_module, self.config['train']['loss'])
        # metrics = getattr(net_module, self.config['metrics'])(num_classes)

        trainable_params = filter(lambda p: p.requires_grad, self.network.parameters())

        opt_name = self.config['optimizer']['type']
        opt_args = self.config['optimizer']['args']
        self.optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)

        # lr_name = self.config['lr_scheduler']['type']
        # lr_args = self.config['lr_scheduler']['args']
        # if lr_name == 'None':
        #     lr_scheduler = None
        # else:
        #     lr_scheduler = getattr(torch.optim.lr_scheduler, lr_name)(self.optimizer, **lr_args)

    def neural_predicate(self, network, path, in_training=True, versions=3):
        data = load_audio(str(path)[1:-1])

        if in_training:
            sig_t, sr, _ = self.t_transforms.apply(data, None)
        else:
            sig_t, sr, _ = self.v_transforms.apply(data, None)

        # print(path)

        length = torch.tensor(sig_t.size(0))
        sr = torch.tensor(sr)
        data = [d.unsqueeze(0) for d in [sig_t, length, sr]]
        try:
            out_raw = network.net(data)
        except RuntimeError:
            print(path)
            print(data)
            raise

        return out_raw.squeeze(0)
