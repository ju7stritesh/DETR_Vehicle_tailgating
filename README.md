Here’s a README in GitHub-friendly Markdown, based **only** on what’s in your repo.

---

# Vehicle Tailgating solution using Deep Sort and DETR

This repository provides a way to **detect and track vehicles** using a **transformer-based object detector (DETR)** and a **tracking algorithm (Deep SORT)** to identify **tailgating** behavior. ([GitHub][1])

> DETR reference: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr) ([GitHub][1])

---

## Repository Structure

* `detr_deep_sort_vehicle.py` — Core logic for detecting and tracking vehicles; includes an option to **draw Region of Interest (ROI)**. ([GitHub][1])
* `inference_tailgating.py` — **Inference entry point**; accepts a video path in the `event()` function and applies tailgating logic. ([GitHub][1])
* `draw_roi.py`, `generate_detections.py`, `features_from_boxes.py` — Utilities used by the pipeline. ([GitHub][1])
* `deep_sort/` — Deep SORT related code. ([GitHub][1])
* `configs/`, `model_data/` — Configuration and model-related assets. ([GitHub][1])
* `Video/`, `NewVideos/` — Sample/input videos (folder names). ([GitHub][1])
* `VideoResults/` — Output videos/results (folder name). ([GitHub][1])
* `requirements.txt` — Python dependencies list. ([GitHub][1])

A sample output video is linked as **`[tailgating.avi](VideoResults%2Ftailgating.avi)
`**.

---

## How it Works (High Level)

1. **DETR** detects vehicles frame-by-frame.
2. **Deep SORT** assigns persistent IDs (ReID) and tracks vehicles.
3. A **tailgating heuristic** evaluates how long a vehicle stays in a defined **ROI** and related conditions to decide violations. ([GitHub][1])

---

## Key Parameters

Configured within the **`Tailgating`** class (in `inference_tailgating.py`) and can be **changed per camera** when calling `event()`:

* `ROI_OVERLAP_THRESHOLD` — Overlap with ROI, **default `0.8`**
* `stop_second_threshold` — Stoppage time inside ROI, **default `4` seconds**
* `Similarity` — Similarity between vehicles (ReID), **default `0.7`**
* `max_dist_covered` — Max distance covered by the same vehicle, **default `800`**
* `min_dist` — Minimum distance a vehicle travels inside ROI, **default `144`** ([GitHub][1])

---

## Steps to Run

1. Run **`inference_tailgating.py`**.
2. The **first frame** of the video will pop up.
3. **Select an ROI** by dragging the mouse from **top-left** to **bottom-right**.
4. Press **`q`** to continue.
5. Vehicles are detected and tracked with IDs.
6. Tailgating is inferred based on **how long a vehicle remained in the ROI** or **was behind another vehicle when a gate was opened**.
7. **Console output** shows the final result. ([GitHub][1])

---

## Files of Interest

* **`inference_tailgating.py`** — main inference & tailgating logic entry. ([GitHub][1])
* **`detr_deep_sort_vehicle.py`** — DETR + Deep SORT integration and ROI drawing support. ([GitHub][1])
* **`README.md`** — original short guide in the repo. ([GitHub][1])
  
#### The results produced from the same can be viewed here
[tailgating.avi](VideoResults%2Ftailgating.avi)


---

## Author

`ju7stritesh@gmail.com` ([GitHub][1])

---

## About

> “This solution provides a way to track vehicles using transformer based Object detection and Tracking algorithm created by me for tailgating.” ([GitHub][1])

---

*This README mirrors the information provided in the repository without adding external details.*

[1]: https://github.com/ju7stritesh/DETR_Vehicle_tailgating "GitHub - ju7stritesh/DETR_Vehicle_tailgating: This solution provides a way to track vehicles using transformer based Object detection and Tracking algorithm created by me for tailgating"

