import numpy as np
import nilearn
import pandas as pd
import os
import json
from modeling_utils import *

data_paths = json.load(open('data_paths.json', 'r'))
participants = prepare_data_csv(data_paths, diag_mapping = diagnosis_mapping)

aggregate_data(participants, '/well/margulies/users/cpy397/SCZ/modeling/data_types.txt')