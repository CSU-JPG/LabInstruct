"""LabInstruct benchmark harness.

Unified generation pipeline for situated instructional video generation
(image-to-video) models, plus the shared QA/judging machinery.

Design:  unified I/O contract, per-model environments, one dispatcher.

The import package is still named ``bench`` so the released code stays
byte-identical to the research code it was extracted from; only the
distribution name is ``labinstruct``.
"""

__version__ = "0.1.0"

# Root of the LabInstruct project (contains data/, bench/, scripts/).
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
