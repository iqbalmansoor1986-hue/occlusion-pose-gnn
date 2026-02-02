# Occlusion-Robust Human Pose Estimation with Graph-Based Refinement

This repository provides a **from-scratch PyTorch implementation** of an occlusion-robust 2D human pose estimation pipeline that formulates pose inference under partial visibility as a **structured restoration** problem over a human-body graph.

The approach integrates:
- **Occlusion-aware initial pose estimation** via heatmaps and **joint-wise visibility prediction**
- **Graph Neural Network (GNN) structural reasoning** over a fixed skeletal topology to refine keypoints under occlusion
- **Geometric consistency regularization** using **bone-length** and **joint-angle feasibility** constraints

The code is designed for clarity and reproducibility and is implemented as a single end-to-end notebook.

---

## Method Summary

Given an input person crop \( I \), the pipeline performs:

1. **Backbone + Heatmaps**
   - A CNN backbone extracts stride-4 feature maps.
   - A heatmap head predicts joint heatmaps \( \{H_i\} \).
   - Coarse keypoints \( \tilde{k}_i \) are obtained using **Soft-Argmax**.

2. **Visibility Prediction**
   - ROI-pooled features \( f_i \) are sampled at predicted keypoints.
   - A visibility head outputs **visibility logits** \( \hat{v}_i \) (converted to probabilities for gating).

3. **Graph-Based Structural Reasoning**
   - The human skeleton is modeled as a graph \( G=(V,E) \) with joints as nodes and anatomical connections as edges.
   - Node embeddings combine position, visibility, and local appearance:
     \[
     $$h_i^{(0)} = [\tilde{k}_i \;\Vert\; \hat{v}_i \;\Vert\; f_i]$$
     \]
   - A multi-layer GNN performs message passing to infer occluded joints from visible, anatomically linked evidence.

4. **Regression Head**
   - A final regression head predicts refined keypoint coordinates \( k_i \) from the refined node embeddings.

---

## Training Objective

The model is trained using a composite loss:
\[
$$\mathcal{L}_{\text{total}}$$ =
$$\mathcal{L}_{\text{hm}} +
\mathcal{L}_{\text{vis}} +
\lambda_1 \mathcal{L}_{\text{bone}} +
\lambda_2 \mathcal{L}_{\text{angle}} +
\lambda_3 \mathcal{L}_{\text{gnn}}$$
\]


Where:
- $$\( \mathcal{L}_{\text{hm}} \)$$: heatmap regression loss
- $$\( \mathcal{L}_{\text{vis}} \)$$: visibility prediction loss (BCEWithLogits)
- $$\( \mathcal{L}_{\text{gnn}} \)$$: coordinate regression loss on visible joints
- $$\( \mathcal{L}_{\text{bone}} \)$$: bone-length consistency regularizer
- $$\( \mathcal{L}_{\text{angle}} \)$$: joint-angle feasibility regularizer
$$

Default weights (as used in the notebook):
- $$\( \lambda_1 = 0.4 \), \( \lambda_2 = 0.2 \), \( \lambda_3 = 1.0 \)$$

---

## Dataset

This repository uses the **COCO 2017 Keypoints** dataset:
- Training: **train2017** + `person_keypoints_train2017.json`
- Validation: **val2017** + `person_keypoints_val2017.json`

The notebook includes download and extraction commands using `curl` (with SSL workaround where needed).

---

## Evaluation

Two evaluation modes are provided:

### 1) PCK (debug metric)
- Computes **PCK@0.05** and **PCK@0.10** using a bbox-based scale proxy
- Evaluated on **visible joints only** (where COCO v==2)

### 2) COCO OKS AP (official metric)
- Exports predictions to COCO-format JSON:
  - `image_id`, `category_id`, `keypoints`, `score`
- Runs `pycocotools.cocoeval.COCOeval` with `iouType="keypoints"`
- Supports top-down evaluation by mapping predictions back to original image coordinates using the inverse affine transform

> Note: This notebook follows a **top-down pose** setting evaluated per COCO person instance (oracle person crop per annotation), not an end-to-end detector + pose system.

---

## Repository Contents

- `Machine_Learning_and_Pattern_Recognition_version2.ipynb`  
  End-to-end implementation:
  - Step 1: Environment setup + COCO download (train/val)
  - Step 2: Dataset + heatmap targets + DataLoaders
  - Step 3: Backbone + heatmaps + soft-argmax + ROI + visibility
  - Step 4: GNN pose refiner (message passing + edge embeddings)
  - Step 5: Loss functions (heatmap, visibility, GNN coord, bone, angle)
  - Step 6: Training loop (AMP-safe)
  - Step 7: Visualization (pred vs GT + visibility)
  - Step 8: PCK evaluation
  - Step 9: COCO OKS AP evaluation (COCOeval)

Outputs (including JSON predictions) are written under:
- `/content/occlusion_pose_gnn/outputs/`

---

## Requirements

Tested in Google Colab with:
- Python 3.x
- PyTorch + CUDA (optional)
- `pycocotools`
- `opencv-python`
- `numpy`
- `matplotlib`
- `tqdm`
- `einops`

---

## Notes / Limitations

- The provided backbone is intentionally lightweight for clarity and from-scratch implementation.
  - Integrating **HRNet-W48** or **ViTPose-S** is possible as an extension.
- Early training (few epochs) may yield near-zero COCO AP; meaningful AP requires longer training and stronger backbones.
- Joint-angle constraints act as plausibility regularizers; rare non-standard articulations are not explicitly modeled.

---

## License

Add your preferred license (e.g., MIT) in `LICENSE`.

---

## Keywords

occlusion, human pose estimation, keypoints, GNN, graph neural network, structural priors, bone-length constraint, joint-angle feasibility, COCO, PyTorch
