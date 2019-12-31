# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-19 20:19:59
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-31 10:49:25

from tabulate import tabulate
import networkx as nx
import json
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import os

from cogdl.models import Build_model
from para_config import Config
from cogdl.tasks import Build_task
from cogdl.datasets import Build_dataset
from cogdl.data import get_dir


if __name__ == '__main__':

    args = Config()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

    random.seed(args.seed[0])
    np.random.seed(args.seed[0])
    args.set_model('gcn')
    args.set_dataset('cora')
    args.set_task('node_classification')

    task = Build_task(args).build()
    result = task.train()
    print(result)

    result_file = get_dir(osp.join(args.save_dir, args.dataset + '_' + args.model + '.json'))
    json_str = json.dumps(result)
    with open(result_file, 'w') as json_file:
        json_file.write(json_str)
