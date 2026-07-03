How each file works:

Here's the rundown, file by file:

1. progress.py 

— Rewritten. emit() now accepts stage, ui_message, meta in addition to message. Instead of storing flat log strings, each call appends a structured dict: {type, stage, text, meta, timestamp}. Added emit_stage_complete(), emit_done(), emit_error() as explicit signals instead of string-matching "DONE"/"ERROR" in text. Terminal print() behavior is untouched.




2. app.py 

— /stream/{job_id} now sends json.dumps(event) instead of raw text, so the frontend gets real structured data. run_pipeline() calls progress.emit_stage_complete() after each of the 3 steps finishes, then emit_done()/emit_error() at the end. (Also carries forward everything from before: UTF-8 file read, upload validation via video_preprocess, duration/resolution rejection handling.)




3. video_preprocess.py 

— Its 3 emit() calls (downscale-start, downscale-done, already-720p) now tag stage="preparing" and pass a short ui_message instead of the verbose terminal text.




4. step_1_functions_segmentation.py 

— All 11 emit() calls (video-open errors, per-metric analysis, per-segment found, plot saved, Excel merge/save messages) tagged stage="segmentation" with clean UI copy.




5. step_1_segmentation.py 

— All 8 emit() calls (video count, per-video scan progress, skip-video, section separators, summary, final "complete") tagged stage="segmentation". The two pure ==== separator lines now pass ui_message="" so they print to terminal but don't show in the UI at all.




6. step_2_functions_video_slice_excel_timestamp.py

 — All 6 emit() calls (open/frame errors, per-clip slicing, Excel save) tagged stage="slicing".




7. step_2_video_slice_excel_timestamp.py 
— All 5 emit() calls (video count, per-asset processing, no-transitions-found, pipeline complete) tagged stage="slicing".




8. step_3_functions_yoloXs_images.py 

— All 7 emit() calls tagged stage="detection". Two functional fixes carried in here too: (a) save_vehicle_registry() now always writes the xlsx even with zero trucks (headers-only), fixing the earlier /download 404; (b) the per-truck log line in compile_segment_database() now also passes meta={"truck_found": True} so the frontend can increment a live counter without parsing text.




9. step_3_yoloXs_images.py

 — All 4 emit() calls (model loading, segment count, per-segment scan) tagged stage="detection".




10. index.html 

— Full rebuild, not a patch:
New visual system (dark navy/asphalt background, amber accent, teal for "done", Space Grotesk + IBM Plex fonts) instead of the black-terminal-with-green-text look.
Four distinct views swapped via JS (idle → processing → done/error), instead of one static page with a log box.
Idle: real drag-and-drop zone + file picker, disabled "Start Processing" until a file's chosen.
Processing: big current-stage headline + description that live-updates from SSE stage_update events, plus a 3-dot trail (Segmenting → Slicing → Detecting) connected by an animated dashed line that lights up as stages complete — driven by stage_complete events.
Done: shows a live truck count (from meta.truck_found events), download button, "process another video" button.
Error: shows the error event's text with a retry button.
Handles upload rejection (too long/wrong resolution) inline instead of a terminal-style error dump.
Accessibility: prefers-reduced-motion respected, visible focus rings, responsive down to mobile.




Net effect: the terminal (what you see when running uvicorn locally) prints exactly what it always did — nothing changed there. The browser now gets a real narrative UI instead of a scrolling log, and the mid-processing-upload / empty-registry-download bugs from earlier are still fixed underneath it.



# 🛠️ MODULE FUNCTIONS BREAKDOWN: segment_pipeline_lib.py

### 1. `compute_histogram`
* **Syntax:** `compute_histogram(frame, bins=64)`
* **Core Description:** This function converts a standard color video frame into a 1D grayscale intensity distribution map. It normalizes pixel distribution counts so that frame-to-frame content can be compared reliably, regardless of minor fluctuating brightness.
* **Inputs (2 total):**
    * `frame`: `numpy.ndarray` | Dimensions: 3D Array (Height x Width x Channels) representing a single BGR color image.
    * `bins`: `int` | Dimensions: Scalar value (Default: `64`). Specifies the discretization resolution of the output array.
* **Outputs (1 total):**
    * `hist`: `numpy.ndarray` | Dimensions: 1D Array with a shape of `(bins,)` (e.g., `(64,)`), containing normalized floating-point probability values between `0.0` and `1.0`.

---

### 2. `extract_video_metrics`
* **Syntax:** `extract_video_metrics(video_path, active_smoothing_method="Savitzky-Golay", savgol_window=7, savgol_poly=3, bins=64)`
* **Core Description:** This function performs the structural frame extraction at a stable rate of 1 frame per second and computes sequential scene changes. It generates mathematical tracking vectors for both Bhattacharyya and Cosine matrix transitions, applying a temporal smoothing filter over the raw signals.
* **Inputs (5 total):**
    * `video_path`: `str` | Dimensions: Scalar path string pointing to the source `.mp4` video file location.
    * `active_smoothing_method`: `str` | Dimensions: Scalar configuration string (e.g., `"Savitzky-Golay"`).
    * `savgol_window`: `int` | Dimensions: Scalar odd integer (Default: `7`) defining the signal window size.
    * `savgol_poly`: `int` | Dimensions: Scalar integer (Default: `3`) specifying the polynomial order.
    * `bins`: `int` | Dimensions: Scalar integer (Default: `64`) passed downstream to the histogram calculator.
* **Outputs (1 tuple containing 3 elements OR `None` if invalid):**
    * `timestamps`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` containing float second metrics for every processed frame.
    * `metrics_raw`: `dict` | Dimensions: Dictionary containing 2 key-value pairs (`"Bhattacharyya"`, `"Cosine"`). Each value points to a 1D `numpy.ndarray` of shape `(N,)` tracking raw variances.
    * `metrics_filtered`: `dict` | Dimensions: Dictionary containing 2 key-value pairs mirroring `metrics_raw`, where each value points to a smoothed 1D `numpy.ndarray` of shape `(N,)`.

---

### 3. `scale_and_segment`
* **Syntax:** `scale_and_segment(filtered_array, raw_array, noise_floor, peak_prominence_factor, min_peak_height, min_peak_distance_secs, base_slope_cutoff)`
* **Core Description:** This function normalizes the filtered time-series data tracks onto a rigid 0 to 1 scale using a defensive noise flooring guard. It runs peak locating routines, executes front/back boundary edge recoveries, and tracks peak slope paths downward to isolate event endpoints.
* **Inputs (7 total):**
    * `filtered_array`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` holding the smoothed change metrics.
    * `raw_array`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` holding the raw change metrics.
    * `noise_floor`: `float` | Dimensions: Scalar configuration float serving as a minimum division cap.
    * `peak_prominence_factor`: `float` | Dimensions: Scalar scaling multiplier to verify target signal prominence.
    * `min_peak_height`: `float` | Dimensions: Scalar threshold value defining the absolute minimum height for peaks.
    * `min_peak_distance_secs`: `int` or `float` | Dimensions: Scalar specifying the spatial window spacing between peaks.
    * `base_slope_cutoff`: `float` | Dimensions: Scalar coefficient tracking the threshold cutoff point above valley floors.
* **Outputs (1 tuple containing 3 elements):**
    * `scaled_smoothed`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` scaled between `0.0` and `1.0`.
    * `scaled_raw`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` scaled between `0.0` and `1.0`.
    * `segments`: `list` | Dimensions: A 1D list containing `M` structured dictionaries (one for each detected video event), where each dict contains three scalar keys: `'start_idx'`, `'peak_idx'`, and `'end_idx'`.

---

### 4. `log_segments`
* **Syntax:** `log_segments(v_name, metric_name, segments, timestamps)`
* **Core Description:** This function outputs timeline details directly to the interactive monitoring terminal for debugging purposes. It formats entry/exit rows specifically for the `"Cosine"` tracking layer to completely prevent downstream database logging duplication.
* **Inputs (4 total):**
    * `v_name`: `str` | Dimensions: Scalar string representing the source filename.
    * `metric_name`: `str` | Dimensions: Scalar string identifying the active evaluation track (e.g., `"Cosine"` or `"Bhattacharyya"`).
    * `segments`: `list` | Dimensions: A 1D list containing `M` index map dictionaries generated by `scale_and_segment`.
    * `timestamps`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` mapping file index targets to exact elapsed seconds.
* **Outputs (1 total):**
    * `excel_rows`: `list` | Dimensions: A 1D list containing structured metadata tracking row dictionaries. If `metric_name != "Cosine"`, this list returns completely empty `[]`.

---

### 5. `render_segment_plot`
* **Syntax:** `render_segment_plot(v_name, metric_name, timestamps, scaled_raw, scaled_smoothed, segments, color, output_path)`
* **Core Description:** This function processes signal curves into high-fidelity evaluation charts using the Matplotlib library backend. It automatically overlays colored transition line flags, shades background regions, adds boundary margins, and exports a high-resolution `.png` file.
* **Inputs (8 total):**
    * `v_name`: `str` | Dimensions: Scalar string representing the processing asset title.
    * `metric_name`: `str` | Dimensions: Scalar string establishing the graph tracking mode title.
    * `timestamps`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` mapping the horizontal X-axis.
    * `scaled_raw`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` mapping the raw Y-axis coordinates.
    * `scaled_smoothed`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` mapping the filtered Y-axis coordinates.
    * `segments`: `list` | Dimensions: A 1D list containing `M` segment dictionary blocks used to coordinate shading ranges.
    * `color`: `str` | Dimensions: Scalar color hex string configuration variable (e.g., `"#1f77b4"`).
    * `output_path`: `str` | Dimensions: Scalar file path string designating the export path destination.
* **Outputs (None):**
    * Returns nothing. It programmatically saves a plot file to disk and opens an on-screen visual chart window workspace instead.

---

### 6. `save_segment_excel`
* **Syntax:** `save_segment_excel(excel_metadata_list, excel_output_path)`
* **Core Description:** This function acts as a persistent file storage gateway by compiling active dictionary records into a unified data structure. It checks for historical workbooks to safely merge files, executes a strict de-duplication guard using file indices, and updates the final `.xlsx` sheet.
* **Inputs (2 total):**
    * `excel_metadata_list`: `list` | Dimensions: A 1D list containing master dictionaries collected across all processed assets.
    * `excel_output_path`: `str` | Dimensions: Scalar file path destination string pointing to the target spreadsheet.
* **Outputs (None):**
    * Returns nothing. It dynamically transforms runtime structures into binary files saved directly on local disk storage arrays.






# 🛠️ MODULE FUNCTIONS BREAKDOWN: slice_pipeline_lib.py

### 1. `compute_histogram`
* **Syntax:** `compute_histogram(frame, bins=64)`
* **Core Description:** This function converts a standard BGR color video frame into a 1D grayscale intensity distribution map. It normalizes pixel distribution counts so the histogram sums strictly to 1.0, safely preventing divide-by-zero math errors down the line.
* **Inputs (2 total):**
    * `frame`: `numpy.ndarray` | Dimensions: 3D Array (Height x Width x Channels) representing a single BGR color image.
    * `bins`: `int` | Dimensions: Scalar value (Default: `64`). Specifies the discretization resolution of the output array.
* **Outputs (1 total):**
    * `hist`: `numpy.ndarray` | Dimensions: 1D Array of shape `(bins,)` containing normalized floating-point probability values.

---

### 2. `extract_cosine_curve`
* **Syntax:** `extract_cosine_curve(video_path, savgol_window=7, savgol_poly=3, bins=64)`
* **Core Description:** This function opens a target video and extracts exactly 1 frame per second to maximize processing speed. It computes the raw Cosine mathematical distance between consecutive frames and applies a Savitzky-Golay temporal filter to iron out camera flicker into a smooth trend line.
* **Inputs (4 total):**
    * `video_path`: `str` | Dimensions: Scalar string pointing to the source `.mp4` file.
    * `savgol_window`: `int` | Dimensions: Scalar odd integer (Default: `7`) defining the signal window size.
    * `savgol_poly`: `int` | Dimensions: Scalar integer (Default: `3`) specifying the polynomial order.
    * `bins`: `int` | Dimensions: Scalar integer (Default: `64`) passed downstream to the histogram calculator.
* **Outputs (1 tuple containing 3 elements OR `None` if invalid/corrupt):**
    * `timestamps`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` containing float metric seconds.
    * `fps`: `float` | Dimensions: Scalar float representing the video's native playback speed.
    * `cosine_smoothed`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` containing the filtered Cosine distance metrics.

---

### 3. `find_cosine_segments`
* **Syntax:** `find_cosine_segments(cosine_smoothed, noise_floor, peak_prominence_factor, min_peak_height, min_peak_distance_secs, base_slope_cutoff)`
* **Core Description:** This function normalizes the smoothed curve to a rigid 0 to 1 scale and identifies motion spikes using SciPy's peak detector. It contains a specialized "Boundary Recovery" layer for activity near the 0.0s mark and utilizes a saddle-point topographical engine to find exact start and end valleys.
* **Inputs (6 total):**
    * `cosine_smoothed`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` holding the smoothed metrics.
    * `noise_floor`: `float` | Dimensions: Scalar configuration float serving as a minimum division cap.
    * `peak_prominence_factor`: `float` | Dimensions: Scalar multiplier to verify target signal prominence.
    * `min_peak_height`: `float` | Dimensions: Scalar threshold value defining the minimum allowed peak height.
    * `min_peak_distance_secs`: `int` or `float` | Dimensions: Scalar specifying the required spatial window between peaks.
    * `base_slope_cutoff`: `float` | Dimensions: Scalar coefficient tracking the threshold cutoff point above valley floors.
* **Outputs (1 total):**
    * `segments`: `list` | Dimensions: A 1D list containing `M` structured dictionaries (one for each detected video event), containing three integer scalar keys: `'start_idx'`, `'peak_idx'`, and `'end_idx'`.

---

### 4. `slice_physical_mp4`
* **Syntax:** `slice_physical_mp4(input_path, start_sec, end_sec, output_path, fps)`
* **Core Description:** This is a physical file engineering function. It seeks directly to a specific timestamp in the master video, reads the frames individually, and re-encodes them into a brand-new, standalone lightweight `.mp4` file containing only the isolated activity.
* **Inputs (5 total):**
    * `input_path`: `str` | Dimensions: Scalar string pointing to the source video file.
    * `start_sec`: `float` | Dimensions: Scalar float marking the beginning of the action.
    * `end_sec`: `float` | Dimensions: Scalar float marking the end of the action.
    * `output_path`: `str` | Dimensions: Scalar string defining where the new clip will be saved.
    * `fps`: `float` | Dimensions: Scalar float dictating the frame rate of the output file.
* **Outputs (None):**
    * Returns nothing. It physically writes a new `.mp4` binary file to the local storage drive.

---

### 5. `slice_video_segments`
* **Syntax:** `slice_video_segments(v_path, v_name, segments, timestamps, fps, segment_dir)`
* **Core Description:** This function acts as the batch manager for a single video. It creates an isolated subfolder for the specific asset, loops through all identified timeline dictionaries, calls the slicing function to generate physical clips, and formats the timeline metrics into database rows.
* **Inputs (6 total):**
    * `v_path`: `str` | Dimensions: Scalar string representing the full source path.
    * `v_name`: `str` | Dimensions: Scalar string representing the source filename.
    * `segments`: `list` | Dimensions: A 1D list containing `M` index map dictionaries generated by `find_cosine_segments`.
    * `timestamps`: `numpy.ndarray` | Dimensions: 1D Array of shape `(N,)` mapping file index targets to exact elapsed seconds.
    * `fps`: `float` | Dimensions: Scalar float of the video's framerate.
    * `segment_dir`: `str` | Dimensions: Scalar string defining the master output folder path.
* **Outputs (1 total):**
    * `excel_rows`: `list` | Dimensions: A 1D list containing `M` structured metadata tracking row dictionaries (logging Source_Video, Segment_ID, Start_Time, End_Time) formatted for database injection.

---

### 6. `save_segment_excel`
* **Syntax:** `save_segment_excel(excel_metadata_list, excel_path)`
* **Core Description:** This function acts as a persistent Excel storage compiler. It takes the newly generated list of segment metadata, safely loads any existing master `.xlsx` workbook, concatenates the new rows, executes a strict de-duplication override if a video was re-processed, and saves the final file.
* **Inputs (2 total):**
    * `excel_metadata_list`: `list` | Dimensions: A 1D list containing master tracking dictionaries collected across the current run.
    * `excel_path`: `str` | Dimensions: Scalar file path destination string pointing to the target spreadsheet.
* **Outputs (None):**
    * Returns nothing. Dynamically transforms runtime structures into binary `.xlsx` files written directly to disk.







# 🛠️ MODULE FUNCTIONS BREAKDOWN: yolox_pipeline_lib.py

### 1. `ensure_yolox_weights`
* **Syntax:** `ensure_yolox_weights(onnx_path, weights_url)`
* **Core Description:** This function serves as an auto-initialization guard. It checks the local drive for the required pre-compiled YOLOX ONNX weights and automatically downloads them from the official repository if they are missing, ensuring the inference engine never crashes due to missing assets.
* **Inputs (2 total):**
    * `onnx_path`: `str` | Dimensions: Scalar string defining the local target path for the `.onnx` model file.
    * `weights_url`: `str` | Dimensions: Scalar string containing the remote URL to download the model from.
* **Outputs (None):**
    * Returns nothing. It streams binary data over the network and writes a physical `.onnx` file to the local disk.

---

### 2. `load_yolox_session`
* **Syntax:** `load_yolox_session(onnx_path)`
* **Core Description:** This function bridges the gap between the static model file and active execution memory. It instantiates the ONNX Runtime session explicitly forcing a CPU-bound execution provider, and extracts the model's dynamic input node name.
* **Inputs (1 total):**
    * `onnx_path`: `str` | Dimensions: Scalar string pointing to the verified `.onnx` model file.
* **Outputs (1 tuple containing 2 elements):**
    * `ort_session`: `onnxruntime.InferenceSession` | Dimensions: A complex memory-bound ONNX execution object.
    * `input_name`: `str` | Dimensions: Scalar string containing the exact internal tensor label the neural network expects (e.g., `"images"`).

---

### 3. `preprocess`
* **Syntax:** `preprocess(img, input_size)`
* **Core Description:** This function is a strict dimensional enforcement layer. It resizes raw video frames while preserving their true physical aspect ratio (letterboxing), pads empty space with a neutral baseline gray (114), and transposes the color channels from standard OpenCV `HWC` format into the network-compliant `CHW` format.
* **Inputs (2 total):**
    * `img`: `numpy.ndarray` | Dimensions: 3D Array (Height x Width x Channels) representing a raw BGR image.
    * `input_size`: `tuple` | Dimensions: A 2-element integer tuple dictating the rigid model input dimensions (e.g., `(640, 640)`).
* **Outputs (1 tuple containing 2 elements):**
    * `padded`: `numpy.ndarray` | Dimensions: 3D Array reshaped to `(Channels x Height x Width)` and cast to `float32` memory types.
    * `r`: `float` | Dimensions: Scalar scale ratio used to accurately map downstream bounding boxes back to the original video size.

---

### 4. `decode_outputs`
* **Syntax:** `decode_outputs(outputs, img_size, strides=(8, 16, 32))`
* **Core Description:** This function acts as the neural network's translation layer. It takes the raw, multi-scale grid prediction tensors emitted by the YOLOX neural graph and mathematically reconstructs them into absolute pixel-based bounding box coordinates.
* **Inputs (3 total):**
    * `outputs`: `numpy.ndarray` | Dimensions: Multidimensional array containing raw network predictions.
    * `img_size`: `tuple` | Dimensions: A 2-element integer tuple matching the model's input size constraints.
    * `strides`: `tuple` | Dimensions: A 3-element integer tuple (Default: `(8, 16, 32)`) representing the feature pyramid downsampling steps.
* **Outputs (1 total):**
    * `outputs`: `numpy.ndarray` | Dimensions: Modified in-place, returning a multidimensional array where the coordinate offsets have been resolved into absolute spatial locations.

---

### 5. `get_truck_boxes`
* **Syntax:** `get_truck_boxes(outputs, ratio, w, h, score_thr, nms_thr, truck_class_id=7)`
* **Core Description:** This function filters the decoded model outputs to strictly isolate target classes (like Trucks). It applies a confidence mask, executes Non-Maximum Suppression (NMS) to delete duplicate overlapping detection boxes, and clips target parameters to prevent fatal edge-boundary errors.
* **Inputs (7 total):**
    * `outputs`: `numpy.ndarray` | Dimensions: Decoded coordinate prediction arrays.
    * `ratio`: `float` | Dimensions: Scalar resize mapping ratio inherited from `preprocess`.
    * `w`: `int` | Dimensions: Scalar native frame width.
    * `h`: `int` | Dimensions: Scalar native frame height.
    * `score_thr`: `float` | Dimensions: Scalar minimum confidence percentage.
    * `nms_thr`: `float` | Dimensions: Scalar geometric intersection limit.
    * `truck_class_id`: `int` | Dimensions: Scalar class index integer (Default: `7` for COCO trucks).
* **Outputs (1 total):**
    * `final_boxes`: `list` | Dimensions: A 2D list array of shape `(M, 4)` containing `M` valid vehicles, where each entry is `[x1, y1, x2, y2]`.

---

### 6. `LightweightTracker`
* **Syntax:** `LightweightTracker(max_disappeared, distance_threshold)` -> `.update(boxes)`
* **Core Description:** This is an instantiated class acting as a real-time Euclidean spatial tracker. It calculates the centroid distance between historical object states and newly detected bounding boxes, locking IDs across frames and automatically recovering tracks that briefly disappear behind environmental obstacles (like trees).
* **Inputs (for `__init__` / 2 total):**
    * `max_disappeared`: `int` | Dimensions: Scalar integer dictating the maximum allowed blind frames before a track is permanently deleted.
    * `distance_threshold`: `float` | Dimensions: Scalar spatial limit denoting how far an object is allowed to physically jump between frames.
* **Inputs (for `.update()` / 1 total):**
    * `boxes`: `list` | Dimensions: A 2D list of current frame bounding coordinate arrays.
* **Outputs (1 total):**
    * `active_tracks`: `dict` | Dimensions: A 1D tracking dictionary where the keys are unique integer IDs (e.g., `1`, `2`) mapping to a list of integer bounding coordinates `[x1, y1, x2, y2]`.

---

### 7. `run_detection_tracking`
* **Syntax:** `run_detection_tracking(local_video_path, local_out_vid, t_start, ort_session, input_name, input_size, truck_class_id, detection_score_thr, detection_nms_thr, tracker_max_disappeared, tracker_distance_thr, collision_std_guard, plate_margin_width_clip, plate_bottom_height_clip)`
* **Core Description:** This is the master per-segment computational engine. It iterates through the video slice frame-by-frame, pipes data to the ONNX model, updates the tracking logic, dynamically captures "Hero Frame" image crops when the vehicle hits the visual center-point, and superimposes active tracking metadata onto an exported debug video.
* **Inputs (14 total):**
    * *(Mixture of scalar path strings, ONNX session objects, integer configuration thresholds, and float boundary limit multipliers defined in preceding functions).*
* **Outputs (1 total):**
    * `active_state_buffer`: `dict` | Dimensions: A complex, multi-layered tracking dictionary where each key is a `track_id`. Values are nested dictionaries containing: `in_time`, `out_time`, `aspect_ratios` (list), `best_truck_img` (3D numpy array), and `best_plate_img` (3D numpy array).

---

### 8. `merge_broken_tracks`
* **Syntax:** `merge_broken_tracks(active_state_buffer, merge_max_time_gap, merge_max_spatial_gap)`
* **Core Description:** A secondary associative cleanup engine. It scans the raw database buffer for interrupted physical tracks (caused by deep shadow or temporary occlusion) and mathematically stitches them back together by matching adjacent exit/entry timestamps alongside minimal spatial drift.
* **Inputs (3 total):**
    * `active_state_buffer`: `dict` | Dimensions: The complex operational dictionary generated by the core tracker loop.
    * `merge_max_time_gap`: `float` | Dimensions: Scalar limit defining max allowable seconds between a track's disappearance and reappearance.
    * `merge_max_spatial_gap`: `float` | Dimensions: Scalar geometric pixel distance limit.
* **Outputs (1 total):**
    * `active_state_buffer`: `dict` | Dimensions: The heavily mutated dictionary object, returning with consolidated profiles and all secondary fragments deleted.

---

### 9. `compile_segment_database`
* **Syntax:** `compile_segment_database(active_state_buffer, seg_id, master_video_name, asset_dir, min_valid_frames_logged)`
* **Core Description:** This function transitions pipeline data into persistent cold storage. It classifies truck types based on their statistical physical aspect ratios, dumps optimal image crops to the hard drive, and links them via local Excel `HYPERLINK` formulas to prevent cloud-syncing lag.
* **Inputs (5 total):**
    * `active_state_buffer`: `dict` | Dimensions: The cleaned operational tracking dictionary.
    * `seg_id`: `int` or `str` | Dimensions: Scalar identification string for the segment.
    * `master_video_name`: `str` | Dimensions: Scalar string representing the source file.
    * `asset_dir`: `str` | Dimensions: Scalar path destination for physical image `.jpg` exports.
    * `min_valid_frames_logged`: `int` | Dimensions: Scalar baseline to aggressively reject false-positive hallucination blips.
* **Outputs (1 total):**
    * `rows`: `list` | Dimensions: A 1D list containing `M` finalized row dictionaries structured for Pandas injection (featuring keys like `"Vehicle ID"`, `"In Time"`, `"Plate File"`, etc.).

---

### 10. `save_vehicle_registry`
* **Syntax:** `save_vehicle_registry(final_database, asset_dir)`
* **Core Description:** The terminal database compilation layer. It structures the global memory lists into an interactive Pandas DataFrame, sorts events chronologically by the master source video, and binds the data to a binary `.xlsx` master ledger using the `openpyxl` engine.
* **Inputs (2 total):**
    * `final_database`: `list` | Dimensions: A 1D array of master aggregated dictionary records collected across all pipeline events.
    * `asset_dir`: `str` | Dimensions: Scalar destination path string for the compiled workbook.
* **Outputs (None):**
    * Returns nothing. Creates and dynamically writes to `Vehicle_Registry_Master.xlsx` on the persistent storage drive.








# 🛠️ MODULE FUNCTIONS BREAKDOWN: gemma_prompt_pipeline_lib.py

### 1. `detect_device`
* **Syntax:** `detect_device()`
* **Core Description:** This function acts as a hardware diagnostic probe. It automatically detects if a compatible CUDA GPU is available for processing and explicitly assigns the mathematical precision level (`float16` for fast GPU compute, `float32` for CPU fallbacks) to prevent tensor mismatch crashes.
* **Inputs (0 total):** * Takes no arguments.
* **Outputs (1 tuple containing 2 elements):**
    * `device`: `str` | Dimensions: Scalar string defining the hardware target (e.g., `"cuda"` or `"cpu"`).
    * `dtype`: `torch.dtype` | Dimensions: PyTorch memory format object (e.g., `torch.float16` or `torch.float32`).

---

### 2. `load_gemma_model`
* **Syntax:** `load_gemma_model(model_id, dtype, device_label="cpu")`
* **Core Description:** This function securely fetches the Multimodal Large Language Model weights from local cache or remote repositories. It binds the neural network directly to the detected hardware layer using the `device_map="auto"` distribution logic and handles fatal load errors safely.
* **Inputs (3 total):**
    * `model_id`: `str` | Dimensions: Scalar string identifying the Hugging Face repository tag.
    * `dtype`: `torch.dtype` | Dimensions: PyTorch memory format object passed from `detect_device()`.
    * `device_label`: `str` | Dimensions: Scalar string (Default: `"cpu"`) used for terminal reporting.
* **Outputs (1 tuple containing 2 elements OR `None, None` if failed):**
    * `processor`: `transformers.AutoProcessor` | Dimensions: The complex neural tokenization and image processing object.
    * `model`: `transformers.AutoModelForMultimodalLM` | Dimensions: The massive multi-gigabyte neural network graph loaded into system memory.

---

### 3. `append_to_report`
* **Syntax:** `append_to_report(report_path, text_content)`
* **Core Description:** A continuous writing and backup mechanism. It opens the designated master log document in append mode (`"a"`), ensuring that new AI-generated event descriptions are safely stacked on top of historical data without overwriting the file.
* **Inputs (2 total):**
    * `report_path`: `str` | Dimensions: Scalar string defining the master `.txt` output location.
    * `text_content`: `str` | Dimensions: Scalar string containing the formatted AI evaluation payload.
* **Outputs (None):**
    * Returns nothing. Streams encoded UTF-8 text directly to the local storage drive.

---

### 4. `extract_segment_storyboard`
* **Syntax:** `extract_segment_storyboard(video_path, interval, max_frames=20)`
* **Core Description:** This function maps continuous video into a digestible visual grid for the LLM. It calculates the specific duration of a clip, pulls frames at strict intervals, converts them from OpenCV BGR to native RGB, and executes a critical VRAM defense limit (`max_frames`) to stop the GPU from experiencing an out-of-memory crash.
* **Inputs (3 total):**
    * `video_path`: `str` | Dimensions: Scalar string targeting the physical `.mp4` segment.
    * `interval`: `float` | Dimensions: Scalar float representing the extraction step time in seconds.
    * `max_frames`: `int` | Dimensions: Scalar integer (Default: `20`) acting as the hard ceiling for tensor array allocations.
* **Outputs (1 tuple containing 2 elements):**
    * `images`: `list` | Dimensions: A 1D array containing `N` extracted `PIL.Image` objects (where `N` $\le$ `max_frames`).
    * `duration`: `float` | Dimensions: Scalar float representing the total calculated seconds of the segment.

---

### 5. `run_gemma_inference`
* **Syntax:** `run_gemma_inference(processor, model, snapshot_slideshow, core_prompt, device, dtype)`
* **Core Description:** The master multimodal inference engine. It constructs the chat template, interleaves the visual storyboard with the text prompt, casts the input tensors into the correct precision limits, passes them through the neural network with gradient tracking disabled (`torch.no_grad()`), and decodes the resulting sequence into plain English.
* **Inputs (6 total):**
    * `processor`: `transformers.AutoProcessor` | Dimensions: The initialized model tokenizer.
    * `model`: `transformers.AutoModelForMultimodalLM` | Dimensions: The initialized generative model graph.
    * `snapshot_slideshow`: `list` | Dimensions: A 1D list of `PIL.Image` visual arrays.
    * `core_prompt`: `str` | Dimensions: Scalar string containing your strict evaluation rules.
    * `device`: `str` | Dimensions: Scalar string (e.g., `"cuda"`).
    * `dtype`: `torch.dtype` | Dimensions: PyTorch numeric formatting rules.
* **Outputs (1 total):**
    * `gemma_text`: `str` | Dimensions: A heavily detailed scalar text string containing the AI's chronological analysis of the video segment.

---

### 6. `purge_gpu_memory`
* **Syntax:** `purge_gpu_memory()`
* **Core Description:** A low-level system garbage collection protocol. It forcibly clears orphaned computational graph tensors and flushes the PyTorch CUDA cache allocator, completely resetting the GPU environment so the next video segment processes on a clean, empty slate.
* **Inputs (0 total):**
    * Takes no arguments.
* **Outputs (None):**
    * Returns nothing. Triggers low-level hardware cache flush operations.




