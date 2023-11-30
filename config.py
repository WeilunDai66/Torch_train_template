from easydict import EasyDict as edict
import numpy as np

exp_name = 'pstn'


cfg = edict()

cfg.device = 'cuda:0'

cfg.epochs = 20
cfg.batch_size = 16


cfg.TRAIN_WRITER = 'result/runs/%s' % exp_name

cfg.output_checkpoint = 'result/checkpoint/%s' % exp_name




