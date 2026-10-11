"""Use the standard COCO evaluator without a machine-specific C++ extension."""

from pycocotools.cocoeval import COCOeval as COCOeval_opt

__all__ = ["COCOeval_opt"]
