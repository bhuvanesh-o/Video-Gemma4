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






BY CLAUDE

# Traffic Control Pipeline — Function Reference

This document covers all 4 library files in the pipeline.
Each file contains reusable functions imported by the main orchestrator scripts.

---

## Files Covered

- `segment_pipeline_lib.py` — change-point detection, segmentation, plotting, Excel logging
- `slice_pipeline_lib.py` — cosine-only segmentation + physical video slicing
- `yolox_pipeline_lib.py` — YOLOX truck detection, tracking, crop saving, Excel export
- `gemma_prompt_pipeline_lib.py` — Gemma 4 model loading, frame extraction, LLM inference

---

---

# FILE 1: segment_pipeline_lib.py

Used by `step_1_segmentation.py`. Handles the full change-point detection pipeline —
computing frame histograms, measuring distances, smoothing, detecting segments, plotting,
and saving to Excel.

---

### 1. compute_histogram(frame, bins=64)

Converts a single BGR video frame to grayscale and computes a normalized intensity histogram.
Every frame gets turned into a probability distribution over pixel intensities.
This is the base feature vector used for all frame-to-frame distance comparisons downstream.

**Inputs:**
- `frame` — numpy array, shape (H, W, 3), dtype uint8. A single BGR frame from OpenCV.
- `bins` — int, default 64. Number of bins to divide the 0–255 intensity range into.

**Outputs:**
- Returns a numpy array of shape (64,) or (bins,), dtype float32.
- All values are in range [0.0, 1.0] and sum to approximately 1.0.

---

### 2. extract_video_metrics(video_path, active_smoothing_method="Savitzky-Golay", savgol_window=7, savgol_poly=3, bins=64)

Opens a video file, samples exactly 1 frame per second, computes both Bhattacharyya
and Cosine histogram distances between consecutive frames, then smooths both distance
curves using the selected method. Converts a raw video into two time-series signals
showing how much the scene changes second by second — the input to segmentation.

**Inputs:**
- `video_path` — str. Full path to the source video file.
- `active_smoothing_method` — str, default "Savitzky-Golay". Only Savitzky-Golay is active currently. Bilateral and Top-Hat are commented out.
- `savgol_window` — int, default 7. Must be an odd number. Window size for the Savitzky-Golay filter in samples (seconds).
- `savgol_poly` — int, default 3. Polynomial order for the Savitzky-Golay filter. Must be less than savgol_window.
- `bins` — int, default 64. Number of histogram bins, passed through to compute_histogram.

**Outputs:**
- Returns a tuple of 3 items on success:
  - `timestamps` — numpy array, shape (N,), float64. Time in seconds for each sampled frame. N is approximately equal to the video duration in seconds.
  - `metrics_raw` — dict with 2 keys: "Bhattacharyya" and "Cosine". Each value is a numpy array of shape (N,), float64. These are the raw unsmoothed frame-to-frame distance values. Index 0 is always 0.0 since there is no previous frame to compare to.
  - `metrics_filtered` — dict with 2 keys: "Bhattacharyya" and "Cosine". Each value is a numpy array of shape (N,), float64. These are the smoothed versions of the raw arrays.
- Returns None if the video is corrupt, unopenable, reports 0 FPS, or has fewer than 2 sampled frames.

---

### 3. scale_and_segment(filtered_array, raw_array, noise_floor, peak_prominence_factor, min_peak_height, min_peak_distance_secs, base_slope_cutoff)

Normalizes a distance curve to [0, 1], runs SciPy's peak detector to find activity spikes,
recovers any peaks right at the video edges that find_peaks misses, then walks down each
peak's left and right slopes to find the true segment start and end at the saddle-point valleys.
This is the core segmentation engine — turns a smoothed distance curve into a list of
time-bounded activity windows.

**Inputs:**
- `filtered_array` — numpy array, shape (N,), float64. The smoothed distance curve for one metric.
- `raw_array` — numpy array, shape (N,), float64. The raw unsmoothed distance curve for the same metric. N must match filtered_array.
- `noise_floor` — float, e.g. 0.0005 for Cosine, 0.005 for Bhattacharyya. Minimum divisor to prevent noise amplification on flat or empty videos.
- `peak_prominence_factor` — float, e.g. 0.12. How much a peak must stand out above its neighbors, expressed as a fraction of the total curve range.
- `min_peak_height` — float, e.g. 0.15. Minimum normalized height [0–1] that a peak must reach to be detected at all.
- `min_peak_distance_secs` — int, e.g. 5. Minimum number of samples (seconds) required between two detected peaks. Peaks closer than this get merged.
- `base_slope_cutoff` — float, e.g. 0.20. How far up from the valley floor toward the peak (as a fraction) to place the segment start/end boundary. 0.20 means the boundary is at 20% of the way up from the valley to the peak.

**Outputs:**
- Returns a tuple of 3 items:
  - `scaled_smoothed` — numpy array, shape (N,), float64. The normalized smoothed curve. Values in [0, 1].
  - `scaled_raw` — numpy array, shape (N,), float64. The normalized raw curve. Values in [0, 1].
  - `segments` — list of dicts. Length equals the number of detected peaks. Each dict has 3 keys:
    - `start_idx` — int. Array index where the segment begins (left slope cutoff point).
    - `peak_idx` — int. Array index of the peak itself.
    - `end_idx` — int. Array index where the segment ends (right slope cutoff point).

---

### 4. log_segments(v_name, metric_name, segments, timestamps)

Prints segment timing details to the console for both metrics, but only returns
Excel-ready row dicts when the metric is Cosine. This avoids writing duplicate rows
since both Bhattacharyya and Cosine produce the same segment windows.

**Inputs:**
- `v_name` — str. Source video filename, e.g. "video1.mp4". Used in console output and Excel rows.
- `metric_name` — str. Either "Cosine" or "Bhattacharyya". Controls whether Excel rows are returned.
- `segments` — list of dicts. Direct output from scale_and_segment.
- `timestamps` — numpy array, shape (N,), float64. Direct output from extract_video_metrics.

**Outputs:**
- Returns a list of dicts. Each dict has 4 keys:
  - `Source_Video` — str. The video filename.
  - `Segment_ID` — int. Segment number starting from 1.
  - `Start_Time` — float, rounded to 2 decimal places. Segment start time in seconds.
  - `End_Time` — float, rounded to 2 decimal places. Segment end time in seconds.
- Returns an empty list if metric_name is not "Cosine".
- Also prints each segment's Start, Peak, and End time to the console regardless of metric.

---

### 5. render_segment_plot(v_name, metric_name, timestamps, scaled_raw, scaled_smoothed, segments, color, output_path)

Draws the raw and smoothed distance curves on a matplotlib figure, shades each detected
segment window in grey, and marks the rise, peak, and fall transition points with colored
vertical lines and text labels. Saves the result as a PNG and opens the plot window.
Used to visually verify that segmentation looks correct for each video and metric.

**Inputs:**
- `v_name` — str. Source video filename. Shown in the plot's main title.
- `metric_name` — str. Metric name. Shown in axis labels, subtitle, and legend.
- `timestamps` — numpy array, shape (N,), float64. Time axis values in seconds.
- `scaled_raw` — numpy array, shape (N,), float64. Normalized raw curve from scale_and_segment.
- `scaled_smoothed` — numpy array, shape (N,), float64. Normalized smoothed curve from scale_and_segment.
- `segments` — list of dicts. Direct output from scale_and_segment.
- `color` — str. Hex color code for the metric trace, e.g. "#1f77b4" for Cosine (blue), "#d62728" for Bhattacharyya (red).
- `output_path` — str. Full file path where the PNG will be saved, including filename and .png extension.

**Outputs:**
- Returns nothing.
- Saves a PNG at 150 dpi to output_path.
- Opens a matplotlib plot window on screen. In VS Code this opens in a separate window since there is no inline notebook rendering.

---

### 6. save_segment_excel(excel_metadata_list, excel_output_path)

Merges new segment rows into any existing Excel workbook, de-duplicates by the
combination of (Source_Video, Segment_ID, Start_Time) keeping the latest run,
and writes the cleaned result back to disk. Creates a fresh workbook if none exists.
This is the final persistence step — the output Excel file is what the downstream
YOLOX pipeline reads to know which video segments to process.

**Inputs:**
- `excel_metadata_list` — list of dicts. All segments collected across all videos in this run. Each dict must have: `Source_Video` (str), `Segment_ID` (int), `Start_Time` (float), `End_Time` (float).
- `excel_output_path` — str. Full path to the .xlsx file to create or update, including filename.

**Outputs:**
- Returns nothing.
- Writes or updates the .xlsx file at excel_output_path.
- Prints a success message or warning to the console.

---
---

# FILE 2: slice_pipeline_lib.py

Used by `step_2_video_slice_excel_timestamp.py`. Handles Cosine-only segmentation
and physically cuts the source video into individual segment .mp4 clips saved into
per-video subfolders.

---

### 1. compute_histogram(frame, bins=64)

Identical to the one in segment_pipeline_lib.py. Converts a BGR frame to grayscale
and returns a normalized intensity histogram. Kept here so this library is self-contained
and does not need to import from the other lib.

**Inputs:**
- `frame` — numpy array, shape (H, W, 3), dtype uint8. A single BGR frame from OpenCV.
- `bins` — int, default 64. Number of histogram bins.

**Outputs:**
- Returns a numpy array of shape (bins,), float32. Values in [0.0, 1.0], sums to ~1.0.

---

### 2. extract_cosine_curve(video_path, savgol_window=7, savgol_poly=3, bins=64)

Opens a video, samples exactly 1 frame per second, computes only the Cosine histogram
distance between consecutive frames, and smooths the curve with Savitzky-Golay.
Slimmer version of extract_video_metrics — Cosine only, and also returns the fps
since it's needed for the physical slicing step.

**Inputs:**
- `video_path` — str. Full path to the source video file.
- `savgol_window` — int, default 7. Must be odd. Savitzky-Golay window size in samples.
- `savgol_poly` — int, default 3. Savitzky-Golay polynomial order.
- `bins` — int, default 64. Histogram bins passed to compute_histogram.

**Outputs:**
- Returns a tuple of 3 items on success:
  - `timestamps` — numpy array, shape (N,), float64. Time in seconds per sampled frame.
  - `fps` — float. The video's actual frame rate. Needed downstream for slice_physical_mp4.
  - `cosine_smoothed` — numpy array, shape (N,), float64. Smoothed Cosine distance curve.
- Returns None if the video is corrupt, unopenable, reports 0 FPS, or has fewer than 2 frames.

---

### 3. find_cosine_segments(cosine_smoothed, noise_floor, peak_prominence_factor, min_peak_height, min_peak_distance_secs, base_slope_cutoff)

Same logic as scale_and_segment in segment_pipeline_lib.py but Cosine only and
does not return the scaled curves — just the segment index dicts.
Normalizes, detects peaks, recovers edge peaks, walks slopes to saddle-point valleys.

**Inputs:**
- `cosine_smoothed` — numpy array, shape (N,), float64. Smoothed Cosine distance curve from extract_cosine_curve.
- `noise_floor` — float, e.g. 0.0005. Minimum divisor guard for flat/empty videos.
- `peak_prominence_factor` — float, e.g. 0.12. Minimum peak prominence as fraction of curve range.
- `min_peak_height` — float, e.g. 0.15. Minimum normalized height for a peak to register.
- `min_peak_distance_secs` — int, e.g. 5. Minimum samples between two peaks.
- `base_slope_cutoff` — float, e.g. 0.20. Fraction above valley floor where boundary is placed.

**Outputs:**
- Returns a list of dicts. Each dict has 3 keys:
  - `start_idx` — int. Array index of segment start.
  - `peak_idx` — int. Array index of the peak.
  - `end_idx` — int. Array index of segment end.
- Returns an empty list if no peaks are found.

---

### 4. slice_physical_mp4(input_path, start_sec, end_sec, output_path, fps)

Seeks to start_sec in the source video, reads frames one by one until end_sec,
and writes them out as a standalone .mp4 file. This physically re-encodes the
sub-clip using OpenCV's VideoWriter with the mp4v codec.

**Inputs:**
- `input_path` — str. Full path to the source video to slice from.
- `start_sec` — float. Start time in seconds to seek to before reading.
- `end_sec` — float. End time in seconds at which to stop writing frames.
- `output_path` — str. Full path for the output .mp4 file including filename.
- `fps` — float. Frame rate to use for the output VideoWriter. Should match the source video's fps from extract_cosine_curve.

**Outputs:**
- Returns nothing.
- Writes a .mp4 file to output_path.

---

### 5. slice_video_segments(v_path, v_name, segments, timestamps, fps, segment_dir)

Creates a per-video subfolder inside segment_dir named after the video, then calls
slice_physical_mp4 for each detected segment to cut and save the clip. Also builds
the Excel row dicts for each segment.

**Inputs:**
- `v_path` — str. Full path to the source video file.
- `v_name` — str. Video filename including extension, e.g. "video1.mp4". Used to name the subfolder and for Excel rows.
- `segments` — list of dicts. Direct output from find_cosine_segments.
- `timestamps` — numpy array, shape (N,), float64. Direct output from extract_cosine_curve.
- `fps` — float. Frame rate from extract_cosine_curve. Passed to slice_physical_mp4.
- `segment_dir` — str. Root folder where per-video subfolders will be created, e.g. "D:\Traffic_Control\trial_video_segments".

**Outputs:**
- Returns a list of dicts. Each dict has 4 keys:
  - `Source_Video` — str. The video filename.
  - `Segment_ID` — int. Segment number starting from 1.
  - `Start_Time` — float, rounded to 2 decimal places.
  - `End_Time` — float, rounded to 2 decimal places.
- Also creates the folder structure and writes the .mp4 clips to disk.
- Folder structure created: segment_dir / video_name_without_extension / segment_1.mp4, segment_2.mp4, ...

---

### 6. save_segment_excel(excel_metadata_list, excel_path)

Same logic as save_segment_excel in segment_pipeline_lib.py. Merges new rows into
any existing workbook, de-duplicates by (Source_Video, Segment_ID, Start_Time),
and saves. This version is slightly simpler — no reset_index call.

**Inputs:**
- `excel_metadata_list` — list of dicts. All segment rows collected across all videos. Each dict must have: `Source_Video` (str), `Segment_ID` (int), `Start_Time` (float), `End_Time` (float).
- `excel_path` — str. Full path to the .xlsx file to create or update.

**Outputs:**
- Returns nothing.
- Writes or updates the .xlsx file at excel_path.
- Prints a success message or warning to the console.

---
---

# FILE 3: yolox_pipeline_lib.py

Used by `step_3_yoloXs_images.py`. Handles the full YOLOX truck detection and tracking
pipeline — loading the ONNX model, preprocessing frames, decoding detections, tracking
vehicles across frames, merging broken tracks, saving crops, and exporting to Excel.

---

### 1. ensure_yolox_weights(onnx_path, weights_url)

Checks if the YOLOX-S ONNX weights file exists at onnx_path. If not, downloads it
once from weights_url using urllib. Creates the parent directory if needed.
Prevents manual wget or file management — fully automatic on first run.

**Inputs:**
- `onnx_path` — str. Local path where the .onnx file should exist or will be saved.
- `weights_url` — str. Direct download URL for the YOLOX-S ONNX weights file.

**Outputs:**
- Returns nothing.
- Downloads and saves the .onnx file to onnx_path if it was missing.
- Prints download status to console.

---

### 2. load_yolox_session(onnx_path)

Initializes an ONNX Runtime inference session from the weights file at onnx_path,
using CPU execution. Extracts the input tensor name needed for running inference.

**Inputs:**
- `onnx_path` — str. Full path to the .onnx weights file.

**Outputs:**
- Returns a tuple of 2 items:
  - `ort_session` — onnxruntime.InferenceSession object. The loaded inference session.
  - `input_name` — str. The name of the model's input tensor, e.g. "images". Used in ort_session.run() calls.

---

### 3. preprocess(img, input_size)

Resizes a BGR frame to fit inside input_size using letterbox scaling (preserves aspect
ratio, pads remaining space with gray value 114), then converts from HWC to CHW format
and returns it as a float32 array ready for ONNX inference.

**Inputs:**
- `img` — numpy array, shape (H, W, 3), dtype uint8. A raw BGR frame from OpenCV.
- `input_size` — tuple of 2 ints, e.g. (640, 640). The target height and width the model expects.

**Outputs:**
- Returns a tuple of 2 items:
  - `padded` — numpy array, shape (3, input_size[0], input_size[1]), dtype float32. The preprocessed frame in CHW format ready for the model.
  - `r` — float. The scaling ratio used during resize. Needed to convert model output coordinates back to original pixel coordinates.

---

### 4. decode_outputs(outputs, img_size, strides=(8, 16, 32))

Converts the raw ONNX model output from grid-relative offsets back into actual pixel
coordinates in the input image space. Works across all 3 detection scales (strides 8, 16, 32).

**Inputs:**
- `outputs` — numpy array, shape (1, num_anchors, 85). Raw output tensor from ort_session.run(). The 85 values are: 4 box coords + 1 objectness + 80 class scores.
- `img_size` — tuple of 2 ints, e.g. (640, 640). The input size used during preprocessing.
- `strides` — tuple of ints, default (8, 16, 32). The detection head stride levels. Do not change.

**Outputs:**
- Returns the same outputs array modified in-place, shape (1, num_anchors, 85), dtype float32. Box coordinates (first 4 values) are now in pixel coordinates relative to the input_size image.

---

### 5. get_truck_boxes(outputs, ratio, w, h, score_thr, nms_thr, truck_class_id=7)

Filters the decoded model outputs to only keep detections classified as trucks
(COCO class ID 7) above the confidence threshold, converts their coordinates back
to original video pixel space, then runs Non-Maximum Suppression to remove duplicates.

**Inputs:**
- `outputs` — numpy array, shape (num_anchors, 85). The decoded output from decode_outputs with the batch dimension removed (index [0]).
- `ratio` — float. The scaling ratio from preprocess. Used to map coordinates back to original resolution.
- `w` — int. Original video frame width in pixels.
- `h` — int. Original video frame height in pixels.
- `score_thr` — float, e.g. 0.45. Minimum confidence score to keep a detection.
- `nms_thr` — float, e.g. 0.6. IoU threshold for Non-Maximum Suppression.
- `truck_class_id` — int, default 7. COCO dataset class index for trucks. Do not change.

**Outputs:**
- Returns a list of bounding boxes. Each box is a list of 4 ints: [x1, y1, x2, y2] in original video pixel coordinates, clipped to frame boundaries.
- Returns an empty list if no truck detections pass the threshold.

---

### 6. LightweightTracker (class)

A simple centroid-based multi-object tracker. Assigns persistent IDs to detected trucks
across frames by matching current detections to existing tracks using Euclidean distance
between centroids. Handles temporary disappearances with a countdown timer before dropping a track.

**Constructor inputs:**
- `max_disappeared` — int, e.g. 60. Number of consecutive frames a track can go unmatched before it is deleted.
- `distance_threshold` — int, e.g. 200. Maximum pixel distance between a track's last centroid and a new detection for them to be matched.

**Method: update(boxes)**
- Input: `boxes` — list of [x1, y1, x2, y2] bounding boxes detected in the current frame.
- Output: dict mapping track_id (int) to box [x1, y1, x2, y2] for all tracks that were matched in this frame (disappeared == 0).

---

### 7. run_detection_tracking(local_video_path, local_out_vid, t_start, ort_session, input_name, input_size, truck_class_id, detection_score_thr, detection_nms_thr, tracker_max_disappeared, tracker_distance_thr, collision_std_guard, plate_margin_width_clip, plate_bottom_height_clip)

The main per-segment processing loop. Runs YOLOX inference on every frame, updates
the tracker, builds a profile for each truck (in/out times, best crop images, aspect
ratios), draws bounding boxes and IDs on the frame, and writes the annotated video.
Also extracts a license plate crop region from the bottom of each truck box.

**Inputs:**
- `local_video_path` — str. Path to the local copy of the segment .mp4 to process.
- `local_out_vid` — str. Path where the annotated output .mp4 will be written.
- `t_start` — float. Absolute start time in seconds of this segment in the original source video. Used to compute true timestamps for each frame.
- `ort_session` — onnxruntime.InferenceSession. Loaded from load_yolox_session.
- `input_name` — str. Model input tensor name from load_yolox_session.
- `input_size` — tuple of 2 ints, e.g. (640, 640). Model input dimensions.
- `truck_class_id` — int, default 7. COCO class ID for trucks.
- `detection_score_thr` — float, e.g. 0.45. Confidence threshold for detections.
- `detection_nms_thr` — float, e.g. 0.6. NMS IoU threshold.
- `tracker_max_disappeared` — int, e.g. 60. Max frames before dropping a lost track.
- `tracker_distance_thr` — int, e.g. 200. Max centroid pixel distance for track matching.
- `collision_std_guard` — float, e.g. 30.0. Max allowed standard deviation of box widths over last 3 frames before freezing aspect ratio logging.
- `plate_margin_width_clip` — float, e.g. 0.20. Fraction of box width to crop from left and right for the plate region.
- `plate_bottom_height_clip` — float, e.g. 0.35. Fraction of box height from the bottom to use as the plate region.

**Outputs:**
- Returns `active_state_buffer` — dict mapping track_id (int) to a profile dict. Each profile dict contains:
  - `in_time` — float. Absolute time in seconds when the truck first appeared.
  - `out_time` — float. Absolute time in seconds of the truck's last seen frame.
  - `best_truck_img` — numpy array (H, W, 3) or None. Best BGR crop of the full truck, taken from the frame where it was closest to the optical center.
  - `best_plate_img` — numpy array (H, W, 3) or None. Cropped plate region from the same hero frame.
  - `aspect_ratios` — list of floats. Width/height ratios collected across frames (used for truck type classification).
  - `frames_tracked` — int. Total number of frames this truck was actively detected.
  - `first_box` and `last_box` — list of 4 ints [x1, y1, x2, y2]. Used for spatial merge checks.
- Also writes the annotated .mp4 to local_out_vid.

---

### 8. merge_broken_tracks(active_state_buffer, merge_max_time_gap, merge_max_spatial_gap)

After processing a segment, scans all track pairs chronologically. If two tracks are
close enough in both time (gap between out_time of A and in_time of B) and space
(distance between last box of A and first box of B), they get merged into one track.
Handles cases where a truck briefly disappears behind a pole or shadow and gets a new ID.

**Inputs:**
- `active_state_buffer` — dict. Direct output from run_detection_tracking.
- `merge_max_time_gap` — float, e.g. 2.5. Maximum seconds allowed between track A ending and track B starting for a merge to happen.
- `merge_max_spatial_gap` — float, e.g. 350. Maximum pixel distance between the last position of A and first position of B for a merge to happen.

**Outputs:**
- Returns the same `active_state_buffer` dict, modified in place, with merged track IDs removed and their data absorbed into the surviving track.

---

### 9. compile_segment_database(active_state_buffer, seg_id, master_video_name, asset_dir, min_valid_frames_logged)

Filters out short-lived noise tracks, classifies each truck by median aspect ratio
into one of 3 types (Multi-Axle/Trailer, Heavy Tipper, Small Box), saves the best
truck and plate crop images as .jpg files, generates Excel HYPERLINK formulas pointing
to those local files, and builds the final row dicts for the master registry.

**Inputs:**
- `active_state_buffer` — dict. Output from merge_broken_tracks.
- `seg_id` — int. The segment number from the Excel timestamps sheet. Used to build the Vehicle ID string.
- `master_video_name` — str. Source video filename. Stored in the Excel row for traceability.
- `asset_dir` — str. Root output folder. Crops are saved to asset_dir/truck_crops/ and asset_dir/plate_crops/.
- `min_valid_frames_logged` — int, e.g. 4. Minimum frames a track must have been seen to be logged. Filters out flickers and false positives.

**Outputs:**
- Returns a list of dicts. Each dict is one row in the master Excel sheet with keys:
  - `Vehicle ID` — str, e.g. "TRUCK_3_2".
  - `In Time` — float. Absolute entry time in seconds.
  - `Out Time` — float. Absolute exit time in seconds.
  - `Truck Type` — str. One of "Multi-Axle/Trailer", "Heavy Tipper", or "Small Box".
  - `Tyres` — str. Estimated tyre count range, e.g. "10-14".
  - `Plate File` — str. Excel HYPERLINK formula or "NO_PLATE".
  - `Truck File` — str. Excel HYPERLINK formula or "NO_IMAGE".
  - `Source` — str. The source video filename.
- Also writes .jpg files to disk under asset_dir/truck_crops/ and asset_dir/plate_crops/.

---

### 10. save_vehicle_registry(final_database, asset_dir)

Takes the complete list of vehicle rows from all segments, converts to a DataFrame,
sorts by source video and entry time, and saves to Vehicle_Registry_Master.xlsx
inside asset_dir using the openpyxl engine so Excel HYPERLINK formulas stay executable.

**Inputs:**
- `final_database` — list of dicts. All rows collected from compile_segment_database across all segments.
- `asset_dir` — str. Folder where Vehicle_Registry_Master.xlsx will be saved.

**Outputs:**
- Returns nothing.
- Writes Vehicle_Registry_Master.xlsx to asset_dir.
- Prints a success message or warning to console.

---
---

# FILE 4: gemma_prompt_pipeline_lib.py

Used by `step_4_prompt_document.py`. Handles device detection, loading the Gemma 4
multimodal LLM from HuggingFace, extracting frame storyboards from video segments,
running inference, and cleaning up GPU memory between segments.

---

### 1. detect_device()

Checks whether a CUDA-capable GPU is available. Returns the appropriate device string
and torch dtype for the hardware. Prints a warning if falling back to CPU since
Gemma 4 is very slow on CPU — expect minutes per segment instead of seconds.

**Inputs:**
- None.

**Outputs:**
- Returns a tuple of 2 items:
  - `device` — str. Either "cuda" or "cpu".
  - `dtype` — torch.dtype. Either torch.float16 (GPU) or torch.float32 (CPU).

---

### 2. load_gemma_model(model_id, dtype, device_label="cpu")

Downloads or loads from local HuggingFace cache the Gemma 4 processor and model
weights. Uses device_map="auto" so HuggingFace handles GPU/CPU placement automatically.
Returns (None, None) and prints the error if loading fails instead of crashing the script.

**Inputs:**
- `model_id` — str. HuggingFace model identifier, e.g. "google/gemma-4-e2b-it".
- `dtype` — torch.dtype. From detect_device. Controls whether weights load in float16 or float32.
- `device_label` — str, default "cpu". Just used in the success print message for display.

**Outputs:**
- Returns a tuple of 2 items on success:
  - `processor` — AutoProcessor object. Handles tokenization and image preprocessing for the model.
  - `model` — AutoModelForMultimodalLM object. The loaded Gemma 4 model ready for inference.
- Returns (None, None) on failure. The error is printed to console.

---

### 3. append_to_report(report_path, text_content)

Simple file writer. Opens the report text file in append mode and writes text_content
followed by a newline. Creates the file if it does not exist yet.
Used to save each segment's Gemma inference result to a persistent text log.

**Inputs:**
- `report_path` — str. Full path to the .txt report file to append to.
- `text_content` — str. The text to write, typically the formatted Gemma inference result for one segment.

**Outputs:**
- Returns nothing.
- Appends text_content + newline to the file at report_path.

---

### 4. extract_segment_storyboard(video_path, interval, max_frames=20)

Opens a pre-cut segment video, samples frames at every `interval` seconds, converts
them from BGR to RGB, and returns them as a list of PIL Images ready for the Gemma
model. Caps the total frame count at max_frames to prevent VRAM crashes on long segments.

**Inputs:**
- `video_path` — str. Full path to the segment .mp4 file.
- `interval` — float, e.g. 0.5. How many seconds between each sampled frame. 0.5 = one frame every half second, 1.0 = one per second.
- `max_frames` — int, default 20. Hard cap on total frames extracted. If the interval would produce more than this, frames are redistributed evenly across the segment duration instead.

**Outputs:**
- Returns a tuple of 2 items:
  - `images` — list of PIL.Image objects. Each image is one sampled frame in RGB format. Length is at most max_frames.
  - `duration` — float. Total duration of the segment in seconds.
- Returns (empty list, 0.0) if the video fails to open or has 0 frames.

---

### 5. run_gemma_inference(processor, model, snapshot_slideshow, core_prompt, device, dtype)

Builds the multi-image chat prompt from the storyboard frames and text prompt,
tokenizes and preprocesses everything, runs model.generate() with no_grad, then
decodes only the newly generated tokens (not the input prompt) and returns the
raw response text. Any exception (OOM, bad input, etc.) propagates up to the
caller's try/except so the loop can skip the segment and continue.

**Inputs:**
- `processor` — AutoProcessor object from load_gemma_model.
- `model` — AutoModelForMultimodalLM object from load_gemma_model.
- `snapshot_slideshow` — list of PIL.Image objects from extract_segment_storyboard.
- `core_prompt` — str. The instruction text telling Gemma what to analyze and report.
- `device` — str. "cuda" or "cpu" from detect_device.
- `dtype` — torch.dtype. float16 or float32 from detect_device.

**Outputs:**
- Returns `gemma_text` — str. The raw decoded model response, not yet stripped of whitespace. Call .strip() on it before displaying or saving.
- Raises any exception that occurs during inference so the caller can catch and log it.

---

### 6. purge_gpu_memory()

Runs Python's garbage collector and then clears the CUDA memory cache if a GPU is
available. Called in the finally block after each segment to prevent VRAM from
accumulating across segments and eventually crashing the script with an OOM error.
Input tensors and generated IDs are freed automatically since they live inside
run_gemma_inference's local scope — no manual del needed here.

**Inputs:**
- None.

**Outputs:**
- Returns nothing.
- Calls gc.collect() and torch.cuda.empty_cache() if CUDA is available.