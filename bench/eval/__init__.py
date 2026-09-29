"""Evaluation: task-conditioned QA -> VLM/human verdicts.

This package holds the shared machinery only -- the QA item schema and parsing
(``qa``), and judge-prompt rendering plus answer validation (``vlm_judge``).

  * QA items are drafted by an LLM and verified by human experts, each with an
    importance level (critical / standard / supplementary) and a dimension.
  * Each item receives one verdict: yes / no / unjudgeable.
"""
