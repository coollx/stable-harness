# Reference index

One row per reference: a self-contained folder under `related-work/` holding the paper, its analyses, and (if available) the code.

| Ref | Paper | Type | Code | Summary | Analyses |
|-----|-------|------|------|------------------|----------|
| HCL | *Harness Continual Learning: Continual Adaptation Beyond Model Parameters* ([paper.pdf](related-work/HCL/paper.pdf)) | method | ❌ no public code | Defines Harness Continual Learning (HCL): a frozen foundation model with a mutable harness (task interface, experience memory, inner skills, router) updated by an untrained proposer behind a commit gate requiring current-task improvement, bounded anchor damage, and validity. Evaluated on ALFWorld, Minecraft, textual and multimodal streams; the tolerance sweep shows forgetting rising monotonically while final performance peaks at an intermediate setting. Plasticity is named but never defined or measured; weights, tools, and the update machinery itself are never modified — our named falsifier baseline. [arXiv:2608.19013](https://arxiv.org/abs/2608.19013) | [paper_analysis.md](related-work/HCL/paper_analysis.md) |

## Literature reviews and syntheses

Cross-paper documents live in `lit-review/`, listed here, never as table rows.
