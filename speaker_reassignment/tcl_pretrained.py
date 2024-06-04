"""
Full config:

{
  "dataset_train": [
    "voxceleb1_train_dev",
    "voxceleb2_train"
  ],
  "dataset_dev": "voxceleb1_validation_dev",
  "batch_size": 48,
  "num_speakers": 7196,
  "database_path": "/net/vol/jenkins/jsons/voxceleb_split.json",
  "chunk_size": 64000,
  "trainer": {
    "factory": "padertorch.train.trainer.Trainer",
    "model": {
      "factory": "....SpeakerNetModel",
      "loss": {
        "factory": "....AngularPenaltySMLoss",
        "in_features": 256,
        "out_features": 7196,
        "loss_type": "aam",
        "eps": 1e-07,
        "s": null,
        "m": null,
        "reduce": "mean"
      },
      "speaker_net": {
        "factory": "....ResNet34",
        "in_channels": 1,
        "channels": [
          64,
          128,
          256,
          256
        ],
        "dvec_dim": 256,
        "activation_fn": "relu",
        "norm": "batch",
        "pre_activation": true
      },
      "sampling_rate": 16000
    },
    "storage_dir": ".../yellow_inherent_elephant",
    "optimizer": {
      "factory": "padertorch.train.optimizer.Adam",
      "gradient_clipping": 5,
      "lr": 0.001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "weight_decay": 0,
      "amsgrad": false
    },
    "loss_weights": null,
    "summary_trigger": [
      1000,
      "iteration"
    ],
    "checkpoint_trigger": [
      1000,
      "iteration"
    ],
    "stop_trigger": [
      500000,
      "iteration"
    ],
    "virtual_minibatch_size": 1
  },
  "p_augment": 0.3,
  "p_reverb": 0.4,
  "p_silence": 0,
  "augment_sets": [
    "noise",
    "music",
    "speech",
    "reverb"
  ],
  "seed": 265671249
}
"""

import functools

import numpy as np
import torch

from .dvectors import SpeakerNetModel, ResNet34


class PretrainedModel:
    def __init__(
            self,
            # ckpt='/scratch/hpc-prf-nt2/cbj/cord/d_vector/yellow_inherent_elephant/checkpoints/ckpt_500000.pth',
            ckpt='/scratch/hpc-prf-nt1/cbj/deploy/speaker_reassignment/egs/vmfmm/ckpt_yellow_inherent_elephant_500000.pth',
            consider_mpi=False,
    ):
        mdl_cfg = {
              "factory": SpeakerNetModel,
              # "loss": None,
              "speaker_net": {
                "factory": ResNet34,
                "in_channels": 1,
                "channels": [64, 128, 256, 256],
                "dvec_dim": 256,
                "activation_fn": "relu",
                "norm": "batch",
                "pre_activation": True,
              },
              "sampling_rate": 16000
        }
        self.ckpt = ckpt
        self.model = SpeakerNetModel.from_config(mdl_cfg)
        self.model.load_checkpoint(self.ckpt, consider_mpi=consider_mpi)
        self.model.eval()

    def __call__(self, observation):
        if len(observation.shape) == 1:
            observation = observation[None]
            squeeze = functools.partial(np.squeeze, axis=0)
        elif len(observation.shape) == 2:
            assert observation.shape[0] < 30, ('Batch dim too large', observation.shape)
            squeeze = lambda x: x
        else:
            raise NotImplementedError(observation.shape)

        with torch.no_grad():
            net_in = self.model.prepare_example({
                'audio_data': {'observation': observation},
                'speaker_id': 'dummy',
                'example_id': 'dummy',
            })
            net_out = self.model.speaker_net(
                self.model.example_to_device(net_in['features']),
                [net_in['features'].shape[1]] * net_in['features'].shape[0],
            )
            return squeeze(net_out[0].numpy())
