import copy
import mmcv
import numpy as np
import os.path as osp
import torch
import warnings
import os
import glob
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
from mmcv.utils import print_log
from torch.utils.data import Dataset

from protogcn.smp import auto_mix2
from .pipelines import Compose
from .pose_dataset import PoseDataset
from .builder import DATASETS
import gc
import json

import sys

sys.path.append("/home/laptq/laptq-fs26-shoplifting-detection/scripts")
from autotrain_utils_fs26_ProtoGCN import val_model_fs26_ProtoGCN


@DATASETS.register_module()
class PoseDatasetEvalFnormalEshoplift(PoseDataset):

    def evaluate(
        self,
        results,
        metrics="F1_Fnormal_Eshoplift",
        metric_options=None,
        logger=None,
        **deprecated_kwargs,
    ):
        assert metrics == "F1_Fnormal_Eshoplift", f"Got {metrics}"

        eval_results = OrderedDict()

        msg = f"Evaluating F1 of frame-level normal recall and event-level shoplift recall..."
        if logger is None:
            msg = "\n" + msg
        print_log(msg, logger=logger)

        work_dir = deprecated_kwargs["work_dir"]
        seed = deprecated_kwargs["seed"]
        runner = deprecated_kwargs["runner"]
        other_kwargs = deprecated_kwargs["other_kwargs"]

        runner.save_checkpoint(
            out_dir=work_dir,
            filename_tmpl="latest_for_val.pth",
            create_symlink=False,
        )

        path_model = glob.glob(f"{work_dir}/latest_for_val.pth")
        assert len(path_model) == 1
        path_model = path_model[0]

        _ = val_model_fs26_ProtoGCN(
            path_model=path_model,
            path_run_dir=os.path.dirname(work_dir),
            id_iter=os.path.basename(work_dir),
            seed=seed,
            **other_kwargs,
        )
        metrics = _["metrics"]
        Fnormal = metrics["frame_level_normal_recall"]
        Eshoplift = metrics["event_level_shoplift_recall"]
        F1_Fnormal_Eshoplift = 2 * Fnormal * Eshoplift / (Fnormal + Eshoplift)

        eval_results["F1_Fnormal_Eshoplift"] = F1_Fnormal_Eshoplift
        eval_results.update(metrics)
        metrics["F1_Fnormal_Eshoplift"] = F1_Fnormal_Eshoplift
        log_msg = [
            f"\nF1_Fnormal_Eshoplift\t{eval_results['F1_Fnormal_Eshoplift']:.4f}",
            f"\nFnormal\t{eval_results['frame_level_normal_recall']:.4f}",
            f"\nEshoplift\t{eval_results['event_level_shoplift_recall']:.4f}",
        ]
        log_msg = "".join(log_msg)
        print_log(log_msg, logger=logger)

        if not hasattr(self, "best_metric"):
            self.best_metric = -1
        if F1_Fnormal_Eshoplift > self.best_metric:
            self.best_metric = F1_Fnormal_Eshoplift
            with open(os.path.join(work_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return eval_results
