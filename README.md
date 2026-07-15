# Homography Estimation on Ice Hockey Rinks

EECS 5323 (Computer Vision) course project — York University, Fall 2025.

**Team 3:** Matin Saedi, Xingbang Tang, Micheal Habib, Lion Isakov

## Abstract

Transforming broadcast ice hockey footage into a normalized bird's-eye view is a critical step for advanced sports analytics, enabling automated player tracking and tactical analysis. This is complicated by the feature sparsity of the ice surface, frequent player occlusions, and varying lighting conditions.

This project presents a robust, coarse-to-fine computer vision pipeline for estimating the homography between broadcast camera views and a standard 2D rink template. Rather than relying purely on local feature detection, it uses a retrieval-based initialization strategy — matching the input frame to a pre-labeled reference database via Histogram of Oriented Gradients (HOG) descriptors — followed by adaptive structural feature extraction, Masked SIFT matching, and sub-pixel refinement with the Enhanced Correlation Coefficient (ECC) algorithm.

On a held-out set of NHL playoff frames, the method achieves a mean reprojection error of **1.70 px** (median 1.23 px), outperforming a general-purpose matcher (LightGlue) in scenarios with heavy occlusion and rink symmetry ambiguities.

The full write-up is in [`docs/report.pdf`](docs/report.pdf).

## Pipeline

1. **Reference Frame Retrieval (HOG):** The test frame and every training frame are resized to 64×128 and encoded with `cv2.HOGDescriptor`. The training frame with the smallest Euclidean distance in HOG-feature space is selected as the best match, providing an initial set of ground-truth keypoints to inherit.
2. **Adaptive Structural Feature Extraction:** A bilateral filter suppresses ice-texture noise while preserving edges, CLAHE (applied to the LAB L-channel) corrects uneven arena lighting, and Gaussian adaptive thresholding produces a binary "skeleton" of the rink markings. An ROI mask — built by detecting the yellow kick-plate in HSV and fitting a 2nd-degree polynomial to its boundary via `numpy.polyfit` — suppresses crowd noise while keeping the playing surface. Connected-component filtering removes small blobs and dilation thickens the remaining lines.
3. **Fine Registration:**
   - *Masked SIFT:* SIFT keypoints/descriptors are extracted only within the rink-line mask, matched with a brute-force k-NN matcher (k=2) and Lowe's ratio test, and used to compute a coarse RANSAC homography (`H_init`).
   - *ECC refinement:* `cv2.findTransformECC`, initialized with `H_init`, aligns the binary skeletons of the test and training frames directly for sub-pixel accuracy (`H_refined`).
4. **Final Homography (RHO):** The refined transform projects the training frame's ground-truth keypoints onto the test frame, and `cv2.findHomography` with the `RHO` flag (3.0 px inlier threshold) computes the final mapping from the test frame to the fixed bird's-eye template.

An earlier variant of the retrieval stage (matching on the rink's yellow boundary line rather than HOG) is preserved in `notebooks/yellow_line_matching_pipeline.ipynb` for reference; the HOG-based approach in the report and in `notebooks/HOG_pipeline.ipynb` is the one that shipped.

## Results

Mean reprojection error across the 10-image test set: **1.70 px** (median 1.23, std 1.68, max 14.26). See Table 1 and Figures 3-5 in the report for the per-image breakdown and the comparison against [LightGlue](https://arxiv.org/abs/2306.13643).

## Repository layout

```
notebooks/
  HOG_pipeline.ipynb                   # main pipeline (HOG retrieval -> adaptive features -> masked SIFT -> ECC -> RHO), incl. LightGlue comparison
  yellow_line_matching_pipeline.ipynb  # earlier yellow-line retrieval variant

docs/
  report.pdf                # full project report

data/
  template/                 # bird's-eye rink template, perspective template, and their labeled keypoints
  train/
    images/, points/        # 21 training frames and their manually annotated keypoints
  test/
    images/, points/        # 10 held-out test frames and their annotated keypoints
```

## Dataset

31 frames curated from broadcast footage of the 2014 Stanley Cup Playoffs (21 for training/retrieval, 10 held out for testing), each manually annotated with pixel coordinates for face-off circles, the center red line, goal lines, and other rink landmarks.

> **Note:** the broadcast frames are screenshots from NHL-owned footage, included here for reproducibility of a course project. They are not original captures and are not licensed for redistribution beyond this academic context.

## Running the notebooks

The notebooks expect to be run from the repository root, with the `data/` folder in place (adjust the hardcoded paths at the top of each notebook if you relocate it). Dependencies:

```
pip install opencv-python opencv-contrib-python numpy matplotlib pandas scikit-image
pip install lightglue  # https://github.com/cvg/LightGlue -- used only for the comparison baseline
```

`opencv-contrib-python` is required for `cv2.SIFT_create()`. LightGlue additionally requires PyTorch.

## AI usage

As documented in the report: AI tools assisted with planning feasible approaches, code debugging/optimization during development (Google Gemini), and grammar/readability polishing of the written report (DeepSeek).
