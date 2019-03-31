import torch.nn as nn
from mask_rcnn.utils.feature_pool import FeatureRunningAvg


class CondHead(nn.Module):
    def __init__(self):
        super(CondHead, self).__init__()
        self.cond_fc = nn.Linear(1024, 256)
        self.actor_cond_score = nn.Linear(256, 7)
        self.action_cond_score = nn.Linear(256, 9)
        self.actor_feature_running_avg = FeatureRunningAvg()


    def forward(self, input, condition_on_actor=True):
        pass