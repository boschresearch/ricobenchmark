#copyright (c) 2025 Robert Bosch GmbH, AGPL-3.0

from .cl_evaluator import (inference_on_continual_learning_dataset,
                           continual_learning_metrics,
                           flatten_continual_learning_dict)
from .cl_events import (ContinualLearningEventStorage,
                        get_cl_event_storage,
                        ContinualLearningJSONWriter, 
                        ContinualLearningWAndBWriter)
# from . import methods
from .methods import naive

import sys
sys.modules["detectron2.continual_learning.naive"] = naive
