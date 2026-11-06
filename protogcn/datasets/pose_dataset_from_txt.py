from collections import OrderedDict
import pickle
import random
import sys
import numpy as np
from torch.utils.data import Dataset
import mmcv
from mmcv.utils import print_log

# Compatibility workaround for unpickling arrays saved with NumPy 2.x in NumPy 1.x
if not hasattr(np, '_core'):
    try:
        import numpy.core as _core
        sys.modules['numpy._core'] = _core
        sys.modules['numpy._core.multiarray'] = _core.multiarray
    except (AttributeError, ImportError):
        pass

from ..utils import get_root_logger
from ..core import macro_f1, top_k_accuracy, mean_class_accuracy, recall_macro, harmonic_mean_recall
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class PoseDatasetFromTxt(Dataset):
    """Pose dataset loading skeleton sequence paths from per-class .txt files,
    matching the format used in SkateFormer feeder_from_txt.

    Args:
        pipeline (list[dict | callable]): A sequence of data transforms.
        classes (list[str]): List of class names. Index represents class label.
        data_dict (dict[str, list[str]]): Mapping from class name to list of txt files.
        test_mode (bool): Whether to build test/validation dataset. Default: False.
        preload (bool): Whether to preload all sequences into RAM at initialization. Default: False.
        shuffle (bool): Whether to shuffle the dataset list during initialization. Default: False.
        seed (int): Random seed for shuffling and sampling. Default: 42.
        balance_classes (bool): Whether to rebalance sample counts across classes (train only). Default: False.
        target_balanced_samples (int | None): Target sample count per class when balance_classes=True. Default: None.
        **kwargs: Extra arguments for compatibility.
    """

    def __init__(
        self,
        pipeline,
        classes: list[str],
        data_dict: dict[str, list[str]],
        test_mode=False,
        preload=False,
        shuffle=False,
        seed=42,
        balance_classes=False,
        target_balanced_samples=None,
        **kwargs,
    ):
        super().__init__()
        self.classes = list(classes)
        self.data_dict = data_dict

        self.test_mode = test_mode
        self.preload = preload
        self.shuffle = shuffle
        self.seed = seed
        self.balance_classes = balance_classes
        self.target_balanced_samples = target_balanced_samples
        self.pipeline = Compose(pipeline)

        self.load_data()

    def load_data(self):
        """Parse txt files and build sample lists."""
        self.num_classes = len(self.classes)
        self.file_paths = []
        self.label = []
        class_counts = {}

        for idx, cls_name in enumerate(self.classes):
            txt_files = self.data_dict.get(cls_name, [])
            cls_paths = []
            for txt_file in txt_files:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    paths = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith('#')
                    ]
                cls_paths.extend(paths)

            if self.balance_classes and not self.test_mode and len(cls_paths) > 0:
                assert self.target_balanced_samples is not None, (
                    'target_balanced_samples must be set when balance_classes=True'
                )
                if len(cls_paths) > self.target_balanced_samples:
                    rng = random.Random(self.seed + idx)
                    cls_paths = rng.sample(cls_paths, self.target_balanced_samples)
                elif len(cls_paths) < self.target_balanced_samples:
                    repeat = max(1, round(self.target_balanced_samples / len(cls_paths)))
                    cls_paths = cls_paths * repeat

            self.file_paths.extend(cls_paths)
            self.label.extend([idx] * len(cls_paths))
            class_counts[cls_name] = len(cls_paths)

        if self.shuffle:
            idxs = list(range(len(self.file_paths)))
            rng = random.Random(self.seed)
            rng.shuffle(idxs)
            self.file_paths = [self.file_paths[i] for i in idxs]
            self.label = [self.label[i] for i in idxs]

        self.label = np.array(self.label, dtype=np.int64)

        split_label = 'VAL' if self.test_mode else 'TRAIN'
        logger = get_root_logger()
        logger.info(
            f'[{split_label}] Initialized {len(self.file_paths)} samples across {len(self.classes)} classes.'
        )
        for idx, cls_name in enumerate(self.classes):
            logger.info(f'  [{idx:2d}] {cls_name}: {class_counts.get(cls_name, 0)} samples')

        if self.preload:
            logger.info(f'[{split_label}] Preloading {len(self.file_paths)} samples into memory...')
            self.preloaded_data = [self.load_file(p) for p in self.file_paths]
        else:
            self.preloaded_data = None

    def load_file(self, file_path):
        """Loads a single sequence file .pkl into a (1, T, V, C) float32 numpy array."""
        with open(file_path, 'rb') as f:
            arr = pickle.load(f)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        assert arr.ndim == 3  # (T, V, C)
        arr = arr[np.newaxis, ...] # -> convert to (1, T, V, C) where M=1
        return arr

    def get_data(self, idx):
        if self.preloaded_data is not None:
            return self.preloaded_data[idx].copy()
        return self.load_file(self.file_paths[idx])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        keypoint = self.get_data(idx)
        results = {
            'filename': self.file_paths[idx],
            'label': int(self.label[idx]),
            'keypoint': keypoint,
            'total_frames': keypoint.shape[1],
            'test_mode': self.test_mode,
        }
        return self.pipeline(results)

    @staticmethod
    def dump_results(results, out):
        """Dump evaluation results to json/yaml/pickle."""
        return mmcv.dump(results, out)

    def evaluate(self, results, metrics='macro_f1', metric_options=None, logger=None, **kwargs):
        """Evaluate predictions against ground truth labels."""
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'Length of results ({len(results)}) != dataset length ({len(self)})'
        )

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        eval_results = OrderedDict()
        gt_labels = self.label

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'macro_f1':
                f1 = macro_f1(results, gt_labels, **kwargs)
                eval_results['macro_f1'] = f1
                print_log(f'\nmacro_f1\t{f1:.4f}', logger=logger)
            else:
                raise KeyError(f'PoseDatasetFromTxt chưa hỗ trợ: {metric} (cho đơn giản)')

        return eval_results


DATASETS.register_module(name='PoseFromTxtDataset', module=PoseDatasetFromTxt)
