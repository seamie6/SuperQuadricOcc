from mmcv.runner import Hook
import torch

class NanCheckHook(Hook):

    def after_train_iter(self, runner):
        # print(f"[NanCheckHook] Iter {runner.iter} running")
        model = runner.model
        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                runner.logger.error(f"[NaN/Inf GRAD] {name}")
                raise RuntimeError(f"Non-finite gradient in {name}")
            if p.data is not None and not torch.isfinite(p.data).all():
                runner.logger.error(f"[NaN/Inf PARAM] {name}")
                raise RuntimeError(f"Non-finite parameter in {name}")
