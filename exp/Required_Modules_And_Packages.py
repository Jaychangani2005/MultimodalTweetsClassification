'''
created on 11/07/2020

@author: Ravindra Kumar
'''

# importing necessary modules

# NOTE: fastai (v1) is optional for the modernized Python 3.11 setup.
# Some legacy notebooks/scripts import this file even when they don't actually use fastai.
try:
	from fastai import *
	from fastai.vision import *
	from fastai.text import *
	from fastai.callbacks import *
	from exp.external.Precision_Module import Precision1
	from fastai.basic_train import *
	from fastai.callbacks import SaveModelCallback, EarlyStoppingCallback
except Exception:  # pragma: no cover
	Precision1 = None
	SaveModelCallback = None
	EarlyStoppingCallback = None


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import exp.external.aidrtokenize as aidrtokenize
from sklearn.metrics import classification_report

from pathlib import Path
import os
import torch
import torch.optim as optim
import random
import tarfile
import zipfile

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from functools import partial



import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
try:
	# Prefer PyTorch's AdamW for modern transformers
	from torch.optim import AdamW
except Exception:  # pragma: no cover
	AdamW = None
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import random
