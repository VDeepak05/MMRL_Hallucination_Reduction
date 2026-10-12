# RQ3: Query-Aware Patch-Level Token Reduction Pipeline (Member 1 Evidence Module)

## 📌 Project Overview
This repository contains the implementation, diagnostic tests, and visual validation scripts for the **Member 1 Evidence Module** under Research Question 3 (RQ3). The project focuses on mitigating Vision-Language Model (VLM) hallucinations by introducing a **query-aware, patch-level token reduction pipeline** built on top of the CLIP vision encoder (`openai/clip-vit-base-patch32`). 

Instead of pruning visual tokens blindly or uniformly, our pipeline computes cross-modal semantic similarity between text prompts and image patches, applies Gumbel-Softmax gating, and outputs a strict dual-stream tensor contract for downstream auditing and hallucination mitigation.

---

## 🏗️ Repository Architecture & File Structure
```text
vlm_hallucination_rq3/
│
├── images/                          # Visual input and output artifacts
│   ├── image3.jpg                   # Target test image (e.g., real-world scene / red car)
│   └── selected_patch_highlight.jpg # Annotated visual output with bounding box
│
└── solution/                        # Core implementation and validation code
    ├── dual_stream_builder.py       # Core pipeline: Member1DualStreamPipeline
    ├── run_batch_test.py            # Diagnostic script: Multi-sample stability & scaling test
    └── visualize_saliency.py        # Diagnostic script: 7x7 spatial mapping & bounding box renderer