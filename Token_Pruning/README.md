# CROP Interactive Demo Guide: Question-Aware Visual Token Pruning for VQA

**Project:** CSE Major Project Research Prototype  
**Research Question (RQ1):** *“Can question-aware visual token selection reduce computational cost without sacrificing VQA accuracy?”*  
**Notebook File:** `CROP_Interactive_Demo_FIXED.ipynb`  
**Architecture:** LLaVA-1.5-7B with CROP Inner-LLM Pruning (ILP) at Layer 2

---

## 1. Executive Summary

This prototype provides an interactive, guide-ready demonstration of **CROP (Contextual Region-Oriented Visual Token Pruning)** for Visual Question Answering (VQA).

In standard LLaVA-1.5-7B:
- An input image is converted into **576 visual tokens** (arranged in a 24×24 spatial grid from CLIP ViT-L/14 at 336×336 resolution).
- All 576 tokens are attended to across all 32 transformer decoder layers, creating quadratic self-attention computational overhead.

Under CROP Inner-LLM Pruning (ILP):
- A query-relevant Region of Interest (ROI) is mapped to intersecting token cells in the 24×24 grid.
- At **decoder layer 2**, non-selected visual tokens are masked out from the self-attention matrix.
- On our benchmark street scene (`street_test.jpg`, 6336×9504 px):
  - **Selected / Kept tokens:** `32 / 576` (**5.56%**)
  - **Masked / Pruned tokens:** `544 / 576` (**94.44%**)
  - **Outcome:** The model still produces the correct factual answer:
    > *“The main object inside the highlighted region is a car.”*

---

## 2. Notebook Execution & GPU Requirements

The notebook is divided into safe, sequential cells:

| Cell | Title | Execution Target | GPU Required? | Description |
|:---:|:---|:---:|:---:|:---|
| **1** | Environment Diagnostic | CPU / GPU | **No** | Reports Python interpreter, CUDA availability, VRAM, and package versions. |
| **2** | Setup CROP & Helper | CPU / GPU | **No** | Clones CROP repo, applies 4-bit device-map patch, initializes token mapping helper. |
| **3** | Upload Image & Question | CPU / GPU | **No** | **Local Upload & Input:** Upload local photo and enter question via dynamic prompt. |
| **4** | Automatic Question Grounding | CPU / GPU | **No** | OWL-ViT detects referring object automatically from question text (e.g. dog, person, car). |
| **5** | Token Grid Visualization | CPU / GPU | **No** | Generates side-by-side 3-panel figure: (A) Image+ROI, (B) 576 tokens, (C) 24×24 Keep-Mask. |
| **6** | Numerical Pruning Audit | CPU / GPU | **No** | Prints exact token counts, sparsity %, and architectural explanations. |
| **7** | Model Inference & Result Card | CPU / GPU | **Optional** | **Unified Step:** Runs live LLaVA on CUDA or grounded zero-shot on CPU, rendering card. |
| **8** | Interactive Bounding Box Explorer | CPU / GPU | **No** | Sliders & "Auto-Detect from Question" button to test arbitrary regions in real-time. |
| **9** | Defense Guide & Research Scope | CPU / GPU | **No** | Comprehensive talking points & answers to guide questions for RQ1. |

> **When Colab GPU Quota is Exhausted:**  
> **Cells 1–7 and Cells 9–10 execute 100% on CPU without error.** You do not need GPU access to show the visual token grid, token selection percentage, or the verified historical benchmark answer to your project guide.

---

## 3. How to Interpret the Visualizations

### The 3-Panel Figure (Cell 5):
1. **Panel A — Input Image with ROI Overlay:**
   - Displays the high-resolution image scaled cleanly with a green bounding box around the target region (the car).
2. **Panel B — Before ILP (Full Visual Capacity):**
   - Shows all 576 cells in green. This represents the default LLaVA behavior where all visual tokens remain active across all 32 transformer layers.
3. **Panel C — After ILP (Mask applied at Decoder Layer 2):**
   - Green squares (**32 tokens**) indicate visual tokens that overlap the bounding box and are retained.
   - Red squares (**544 tokens**) indicate visual tokens that are masked out from the attention matrix at layer 2.

---

## 4. Distinguishing Historical Record vs. Fresh Live Inference

The notebook strictly enforces honest evaluation reporting:
- **`[HISTORICAL VALIDATED RESULT]` (Orange Badge in Cell 7):**
  - Confirmed evidence from the validated run on Tesla T4 (PyTorch 2.6.0+cu124, 4-bit quantized LLaVA-1.5-7B, ILP layer 2 mask mode).
  - Displayed whenever running in CPU mode or before live inference is triggered.
  - Record payload:
    ```json
    {
      "prompt": "What is the main object inside the highlighted region?",
      "text": "The main object inside the highlighted region is a car.",
      "model_id": "llava-v1.5-7b",
      "metadata": {"crop_method": "ilp"}
    }
    ```
- **`[LIVE PREDICTION]` (Green Badge in Cell 7):**
  - Only displayed if Cell 8 ran to completion in the current session on an active GPU.

---

## 5. Automatic Question-Aware Region Localization

The prototype includes an **Automatic Question-Aware Localizer** (in Cell 4 and interactive Cell 9):

1. **Natural Language Question Parsing:**
   - Extracts target candidate queries from the question (e.g., *"What is the car doing?"* $\rightarrow$ `"car"`, or *"What are the pedestrians doing?"* $\rightarrow$ `"pedestrian"`).
2. **Open-Vocabulary Zero-Shot Grounding:**
   - Evaluates the image and target phrase using open-vocabulary zero-shot detection (`google/owlvit-base-patch32` / salient foreground region), predicting the bounding box `[x1, y1, x2, y2]`.
3. **Automated Bridge to CROP:**
   - The predicted bounding box is passed directly to `bbox_to_token_indices`, creating the 24×24 token keep-mask without requiring manual coordinate entry.
4. **Interactive Auto-Detection:**
   - Cell 9 includes an **"Auto-Detect from Question"** button allowing guides to type new questions and witness automatic bounding-box and token-grid recalibration in real time.

---

## 6. What the Prototype Proves vs. What It Does Not Prove

### Supported Claims:
1. **Selective Visual Attention:** LLaVA-1.5-7B does not require all 576 visual tokens across all layers to answer a localized visual question.
2. **Pruning Effectiveness:** Retaining only 5.56% of visual tokens at layer 2 allows the model to answer accurately (*"The main object inside the highlighted region is a car."*).
3. **Correct Architectural Integration:** The CROP ILP attention mask applies at decoder layer 2 without disrupting text-token generation or requiring model retraining.

### Unsupported Claims (Do Not Make These):
1. **Do NOT claim a 94.44% total FLOPs reduction:**  
   Pruning 94.44% of visual tokens at layer 2 does not reduce total computation by 94.44%. The vision transformer (CLIP), multimodal projector, and LLM layers 0 and 1 still process all 576 tokens. Furthermore, text tokens are never pruned.
2. **Do NOT claim generic VQA accuracy from a single sample:**  
   A single test case demonstrates prototype feasibility. Establishing statistical accuracy preservation requires benchmarking over standard datasets (e.g., VQAv2, GQA).

---

## 7. How to Run Live Inference (When GPU Quota Resets)

When you have access to a Google Colab GPU runtime:
1. Go to **Runtime -> Change runtime type -> T4 GPU**.
2. Run **Cell 1** to confirm GPU detection (`CUDA available : True`).
3. If packages are missing in the fresh runtime, run:
   ```bash
   !pip install bitsandbytes accelerate transformers shortuuid
   ```
4. Run **Cells 1 through 7**.
5. Run **Cell 8**: It will execute `python -m crop.eval.model_vqa_loader` using `sys.executable` in ~2–3 minutes and generate fresh output.
6. Run **Cell 7** again to see the updated live prediction card.
