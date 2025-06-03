import numpy as np
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class RecognizerGCN(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition."""

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]

        losses = dict()
        x, get_graph = self.extract_feat(keypoint)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze(-1)
        loss = self.cls_head.loss(cls_score, get_graph, gt_label)
        losses.update(loss)

        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        to_numpy = kwargs.get("to_numpy", True)
        return_feat = kwargs.get("return_feat", False)
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc,) + keypoint.shape[2:])

        feat, get_graph = self.extract_feat(keypoint)
        x = feat
        feat_ext = self.test_cfg.get("feat_ext", False)
        pool_opt = self.test_cfg.get("pool_opt", "all")
        score_ext = self.test_cfg.get("score_ext", False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == "all":
                pool_opt == "nmtv"
            if pool_opt != "none":
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, "The shape is N, M, C, T, V"
            if pool_opt != "none":
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum("nmctv,oc->nmotv", x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16) if to_numpy else x

        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        if "average_clips" not in self.test_cfg:
            self.test_cfg["average_clips"] = "prob"

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [(x.data.cpu().numpy() if to_numpy else x) for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        ret = cls_score.data.cpu().numpy() if to_numpy else cls_score
        if return_feat:
            return ret, self.feat_to_emb(feat)
        return ret

    def feat_to_emb(self, x):
        B, M, C, T, V = x.shape
        assert M == 1, "If error, consider .view(B * M, C, T, V)"
        x = x.view(B, C, T, V)
        x = x.mean(dim=(2, 3))
        return x

    def forward(
        self,
        keypoint,
        label=None,
        return_loss=True,
        return_feat=False,
        to_numpy=True,
        already_merged__batchsize_numclips_numpersons=False,
        **kwargs
    ):
        """Define the computation performed at every call."""
        if already_merged__batchsize_numclips_numpersons:
            B, C, T, V = keypoint.shape
            keypoint = keypoint.permute(0, 2, 3, 1).view(B, 1, 1, T, V, C)
        if return_loss:
            if label is None:
                raise ValueError("Label should not be None.")
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, to_numpy=to_numpy, **kwargs)

    def extract_feat(self, keypoint):

        return self.backbone(keypoint)
