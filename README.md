# Video-Gemma4: End-to-End Traffic Video Analytics Pipeline

An end-to-end traffic video analysis system for detecting, tracking, extracting, and describing trucks from uploaded videos.

The pipeline combines traditional video processing, YOLOX-S object detection, OpenVINO-optimized CPU inference, PaddleOCR, and multimodal Gemma models. It is exposed through a FastAPI web application with background processing, live progress updates, downloadable results, and Docker support.

GitHub: https://github.com/bhuvanesh-o/video-gemma4


---


### Mentorship

This project was developed under the guidance and mentorship of:

- **Dr. Kushal Shah** — [LinkedIn Profile](http://linkedin.com/in/kushal-shah-95b9a3b/)
- **Dr. Nilanjan Banerjee** — [LinkedIn Profile](https://www.linkedin.com/in/nilanjanbanerjee/)


---


## Overview

The system takes a traffic video as input and automatically performs:

1. Video validation and preprocessing
2. Temporal activity segmentation
3. Video segment extraction
4. Truck detection and tracking
5. Representative truck image selection
6. License plate region extraction and OCR
7. Vehicle-level multimodal analysis using Gemma
8. Structured Excel report generation
9. Browser preview and downloadable results

The project was also designed to explore CPU inference optimization using OpenVINO, asynchronous inference, FP32/FP16/INT8 model variants, post-training quantization, and resource benchmarking.

---


## Prerequisites

For local execution:

- Python 3.x
- Git
- FFmpeg
- Internet connection for initial model downloads
- Sufficient disk space for model weights and generated video files

Optional:

- Docker and Docker Compose
- OpenRouter API key for cloud Gemma inference
- Hugging Face token if required for model access


---


## Quick Start

### Clone the repository

```bash
git clone https://github.com/bhuvanesh-o/video-gemma4.git
cd video-gemma4
```

### Open in VS Code

```bash
code .
```

### Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Create `.env`

Windows:

```powershell
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```



### Start the application

Recommended:

```bash
python -m uvicorn app:app --reload
```

Alternatively:

```bash
python main.py
```

Open:

```text
http://127.0.0.1:8000
```

Upload a traffic video through the web interface and wait for the pipeline to complete.


---


## Usage

1. Start the FastAPI application.
2. Open `http://127.0.0.1:8000`.
3. Upload a supported traffic video.
4. Wait while the pipeline performs:
   - preprocessing
   - segmentation
   - video slicing
   - truck detection and tracking
   - OCR
   - Gemma-based analysis
5. Preview the detected vehicle information in the browser.
6. View annotated videos and extracted images.
7. Download the generated Excel vehicle registry.


---

## Expected Outputs


After successful processing, each job generates its own directory under:


'''text
storage/jobs/<job_id>/


---

## Pipeline

```text
Uploaded Traffic Video
        │
        ▼
┌──────────────────────────┐
│ Video Validation         │
│ + Resolution Processing  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Step 1: Segmentation     │
│ Histogram-based change   │
│ detection                │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Step 2: Video Slicing    │
│ Extract relevant clips   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Step 3: Detection        │
│ YOLOX-S + OpenVINO       │
│ Tracking + PaddleOCR     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Step 4: Description      │
│ Cloud + Local Gemma      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Vehicle Registry         │
│ Images + Video + Excel   │
└──────────────────────────┘
```

---

# Features

## 1. Video preprocessing

Uploaded videos are validated before the full pipeline begins.

The preprocessing layer handles:

- video metadata extraction
- resolution validation
- automatic downscaling of videos above the target resolution
- rejection of unsupported/invalid inputs
- job-specific storage

Supported upload extensions:

```text
.mp4
.avi
.mov
.mkv
```

The pipeline is currently designed around traffic videos of at least 720p resolution.

---

## 2. Temporal video segmentation

Instead of running the expensive detection pipeline blindly over the entire video, the system first identifies regions containing meaningful scene activity.

The segmentation stage:

- samples video frames
- converts frames to grayscale
- computes normalized image histograms
- compares consecutive frames using:
  - Cosine distance
  - Bhattacharyya distance
- smooths the resulting signals
- detects significant peaks
- estimates segment start/end boundaries
- saves the detected timeline to Excel

Output:

```text
segment_timestamps.xlsx
```

This reduces unnecessary downstream processing.

---

## 3. Video slicing

The detected activity intervals are converted into independent video clips.

Example:

```text
trial_video_segments/
└── source_video/
    ├── segment_1.mp4
    ├── segment_2.mp4
    └── ...
```

These clips are then passed to the detection stage.

---

# 4. YOLOX-S Truck Detection

Truck detection is performed using **YOLOX-S**.

The original model is represented as ONNX and is executed through **OpenVINO** for optimized CPU inference.

The pipeline currently targets the COCO truck class.

Main detection operations include:

- 640 × 640 letterbox preprocessing
- YOLOX output decoding
- confidence filtering
- Non-Maximum Suppression
- truck-only class filtering

---

# 5. OpenVINO Model Optimization

The project supports three inference precision modes:

```text
FP32
FP16
INT8
```

The mode can be configured through `.env`:

```env
MODEL_PRECISION=FP32
```

or:

```env
MODEL_PRECISION=FP16
```

or:

```env
MODEL_PRECISION=INT8
```

---

## FP32

When FP32 is selected:

```text
YOLOX-S ONNX
    ↓
OpenVINO conversion
    ↓
FP32 OpenVINO IR
```

The model is generated automatically if the FP32 OpenVINO model is missing.

Files:

```text
weights/yolox_small_fp32.xml
weights/yolox_small_fp32.bin
```

---

## FP16

When FP16 is selected:

```text
YOLOX-S ONNX
    ↓
OpenVINO conversion
    ↓
FP16-compressed OpenVINO IR
```

Files:

```text
weights/yolox_small_fp16.xml
weights/yolox_small_fp16.bin
```

---

## INT8

INT8 inference uses a post-training quantized OpenVINO model.

Required files:

```text
weights/yolox_small_int8.xml
weights/yolox_small_int8.bin
```

The INT8 model can be generated using the provided calibration and quantization utilities.

Typical workflow:

```bash
python create_model_variants.py
python build_calibration_data.py
python quantize_int8.py
```

The quantization pipeline uses representative video frames and NNCF post-training quantization.

---

# 6. Asynchronous OpenVINO Inference

The detector supports multiple simultaneous OpenVINO inference requests.

For example:

```env
NUM_INFER_REQUESTS=4
```

Frames are submitted asynchronously so inference can overlap rather than processing each frame using a strict:

```text
submit → wait → process → submit → wait
```

cycle.

The implementation still processes completed frames in chronological order because tracking depends on frame ordering.

The OpenVINO performance mode can also be configured:

```env
OPENVINO_PERFORMANCE_HINT=THROUGHPUT
```

or:

```env
OPENVINO_PERFORMANCE_HINT=LATENCY
```

---

# 7. Lightweight Multi-Object Tracking

Detected trucks are assigned persistent IDs across video frames.

The tracker uses spatial/centroid information and handles short detection disappearances.

Example vehicle IDs:

```text
TRUCK_1_1
TRUCK_1_2
TRUCK_2_1
```

Broken tracks may also be merged using temporal and spatial constraints.

---

# 8. Representative Truck Image Selection

Rather than saving every detected frame, the pipeline searches for useful representative images.

Candidate frames are evaluated using factors such as:

- image sharpness
- truck bounding-box size
- distance from image boundaries
- temporal spacing
- approach/receding behaviour

The highest-quality truck crops are saved for later OCR and multimodal analysis.

---

# 9. License Plate Processing with PaddleOCR

The pipeline uses **PaddleOCR** for text detection and recognition.

It attempts to:

1. locate likely text/plate regions
2. score candidate regions
3. run OCR
4. select the most plausible license plate text
5. save the plate crop

Example output directories:

```text
final_assets/
├── truck_crops/
└── plate_crops/
```

PaddleOCR model files are automatically downloaded and cached on first use.

Therefore, the first run may take longer than later runs.

---

# 10. Gemma-Based Vehicle Analysis

Each selected truck image is analyzed using two multimodal model paths.

## Cloud model

A Gemma model is accessed through **OpenRouter**.

It analyzes information such as:

- vehicle type
- color
- number plate
- brand / make
- load / cargo
- visible condition or damage
- additional identifying details

`OPENROUTER_API_KEY` is optional.

If the key is not provided, the cloud Gemma stage is skipped while the remaining pipeline can continue.

---

## Local model

The project also supports local Gemma inference through **LiteRT-LM**.

This allows comparison between:

```text
Cloud multimodal inference
vs.
Local multimodal inference
```

The local model is cached after download/setup.

---

# 11. Structured Vehicle Registry

The final output is stored in:

```text
Vehicle_Registry_Master.xlsx
```

Depending on available detections and model outputs, the registry contains information including:

- Vehicle ID
- entry / exit timing
- truck classification information
- truck image
- plate image
- OCR output
- cloud Gemma analysis
- local Gemma analysis

The Excel output can be downloaded directly from the web interface.

---

# 12. Annotated Video Output

Detected trucks are drawn on output video segments.

OpenCV initially writes the video and FFmpeg is used to re-encode the result into a browser-friendly H.264 format.

FFmpeg therefore needs to be available when running the project natively.

Check with:

```bash
ffmpeg -version
```

If FFmpeg is unavailable, the pipeline falls back to the raw OpenCV video output, although browser playback may not work correctly.

Docker installs FFmpeg automatically.

---

# 13. FastAPI Web Application

The project uses **FastAPI** as its application backend.

The official application entry point is:

```text
app.py
```

For development:

```bash
python -m uvicorn app:app --reload
```

The web interface is then available at:

```text
http://127.0.0.1:8000
```

---

# 14. Background Processing

Video processing is performed as a background task.

The upload flow is:

```text
Browser uploads video
        ↓
Server generates job ID
        ↓
Video is validated
        ↓
Browser immediately receives job ID
        ↓
Full AI pipeline continues in background
```

This prevents the upload request itself from remaining blocked for the entire processing duration.

---

# 15. Live Progress with Server-Sent Events

The frontend receives live pipeline progress through **Server-Sent Events (SSE)**.

This allows the UI to show stages such as:

```text
Segmentation
    ↓
Slicing
    ↓
Detection
    ↓
Description
    ↓
Complete
```

SSE was used because progress communication mainly travels from:

```text
server → browser
```

and therefore does not require the complexity of a full WebSocket connection.

---

# 16. Job-Isolated Storage

Every uploaded video receives a unique job ID.

Example:

```text
45392bdb
```

Files belonging to the job are stored under:

```text
storage/jobs/45392bdb/
```

A typical job directory looks like:

```text
storage/
└── jobs/
    └── <job_id>/
        ├── raw/
        │   └── input_video.mp4
        │
        ├── segment_timestamps.xlsx
        │
        ├── trial_video_segments/
        │   └── ...
        │
        ├── temp/
        │
        └── final_assets/
            ├── Vehicle_Registry_Master.xlsx
            ├── truck_crops/
            ├── plate_crops/
            └── annotated_segments/
```

Job IDs and generated storage paths are validated to reduce path-traversal risks.

Old jobs are automatically cleaned from local storage after the configured retention period.

---

# 17. Upload Safety

Uploaded filenames are sanitized before being written to disk.

The API only accepts configured video extensions:

```text
.mp4
.avi
.mov
.mkv
```

The storage layer also validates:

- job IDs
- relative paths
- storage-path containment

to prevent generated paths from escaping the intended job directory.

---

# 18. Frontend

The browser interface provides:

- drag-and-drop video upload
- pipeline stage visualization
- live progress updates
- vehicle result preview
- truck image preview
- plate image preview
- annotated video access
- Excel result download

The frontend communicates with the FastAPI backend through `/api/...` routes.

---

# Main API Routes

Important API routes include:

```text
POST /api/upload
```

Uploads and validates a video and starts background processing.

```text
GET /api/stream/{job_id}
```

Streams live pipeline progress using Server-Sent Events.

```text
GET /api/preview/{job_id}
```

Returns the generated vehicle registry as browser-readable JSON.

```text
GET /api/download/{job_id}
```

Downloads the generated Excel report.

Additional routes serve generated truck crops, plate images, and annotated videos.

---

# 19. Benchmarking

The repository includes:

```text
benchmark_videos.py
```

for benchmarking the complete pipeline.

The benchmark measures values such as:

- preprocessing time
- segmentation time
- slicing time
- detection time
- description time
- total processing time
- average CPU usage
- peak CPU usage
- average RAM usage
- peak RAM usage
- Docker/container memory when available
- effective detection FPS
- real-time factor

It can also compare different combinations of:

```text
FP32 / FP16 / INT8
```

and:

```text
LATENCY / THROUGHPUT
```

performance hints.

Example concept:

```text
Real-Time Factor =
total processing time / video duration
```

Interpretation:

```text
< 1  : faster than real time
= 1  : real time
> 1  : slower than real time
```

---

# Project Structure

```text
video-gemma4/
│
├── app.py
├── main.py
│
├── routers/
│   ├── __init__.py
│   ├── frontend.py
│   └── pipeline.py
│
├── storage.py
├── progress.py
├── video_preprocess.py
│
├── step_1_segmentation.py
├── step_1_functions_segmentation.py
│
├── step_2_video_slice_excel_timestamp.py
├── step_2_functions_video_slice_excel_timestamp.py
│
├── step_3_yoloXs_images.py
├── step_3_functions_yoloXs_images.py
│
├── step_4_prompt_document.py
├── step_4_functions_prompt_document.py
│
├── benchmark_videos.py
├── resource_monitor.py
│
├── create_model_variants.py
├── build_calibration_data.py
├── quantize_int8.py
│
├── index.html
├── index.css
│
├── weights/
├── storage/
│
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
│
├── Dockerfile
├── docker-compose.yml
│
├── LICENSE
└── README.md
```

---

# Technologies Used

### Machine Learning / Computer Vision

- YOLOX-S
- OpenVINO
- NNCF
- PaddleOCR
- OpenCV
- NumPy
- SciPy

### Multimodal / LLM

- Gemma
- OpenRouter
- LiteRT-LM
- Hugging Face

### Backend

- FastAPI
- Python
- Server-Sent Events
- background tasks

### Data / Reporting

- Pandas
- OpenPyXL
- Excel

### Deployment

- Docker
- Docker Compose
- FFmpeg

---

# Installation

## Option 1 — Docker

Docker is the recommended setup because the project depends on Python packages, FFmpeg, machine-learning runtimes, and model caches.

### 1. Clone the repository

```bash
git clone https://github.com/bhuvanesh-o/video-gemma4.git
cd video-gemma4
```

### 2. Create the environment file

Linux/macOS:

```bash
cp .env.example .env
```

Windows:

```powershell
copy .env.example .env
```

### 3. Edit `.env`

Example:

```env
OPENROUTER_API_KEY=
HF_TOKEN=

MODEL_PRECISION=FP32
OPENVINO_PERFORMANCE_HINT=THROUGHPUT
NUM_INFER_REQUESTS=4
```

### 4. Start the application

```bash
docker compose up --build
```

### 5. Open the web interface

```text
http://localhost:8000
```

---

# Docker Storage

The Docker configuration maps:

```text
./storage  → /app/storage
./weights  → /app/weights
```

Therefore uploaded videos, model files, and generated results remain available on the host machine even if the container is stopped.

Model caches use Docker named volumes:

```text
huggingface_cache
litert_cache
paddle_cache
```

This avoids re-downloading large model files every time the container restarts.

---

# Option 2 — Local Python Setup

### 1. Clone

```bash
git clone https://github.com/bhuvanesh-o/video-gemma4.git
cd video-gemma4
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install FFmpeg

FFmpeg must be available on the system PATH.

Verify:

```bash
ffmpeg -version
```

### 5. Create `.env`

```env
OPENROUTER_API_KEY=
HF_TOKEN=

MODEL_PRECISION=FP32
OPENVINO_PERFORMANCE_HINT=THROUGHPUT
NUM_INFER_REQUESTS=4
```

### 6. Run

```bash
python -m uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

# Environment Variables

| Variable | Example | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | `...` | Enables cloud Gemma analysis |
| `HF_TOKEN` | `...` | Hugging Face model access when required |
| `MODEL_PRECISION` | `FP32` | `FP32`, `FP16`, or `INT8` |
| `OPENVINO_PERFORMANCE_HINT` | `THROUGHPUT` | `THROUGHPUT` or `LATENCY` |
| `NUM_INFER_REQUESTS` | `4` | Number of asynchronous OpenVINO requests |

Never commit the real `.env` file.

Use:

```text
.env.example
```

as the public template.

---

# First Run

The first run may be slower because some model assets need to be downloaded or prepared.

Examples include:

- YOLOX-S model weights
- PaddleOCR detection model
- PaddleOCR recognition model
- local Gemma model assets
- OpenVINO compiled/cache files

Later runs reuse cached files.

An internet connection is therefore recommended for initial setup.

Messages such as:

```text
Model files already exist. Using cached files.
```

from PaddleOCR are normal and indicate that cached model files are being reused.

A Paddle warning about `ccache` is also not required for normal inference.

---

# Running INT8

To use INT8:

```env
MODEL_PRECISION=INT8
```

Ensure these files exist:

```text
weights/yolox_small_int8.xml
weights/yolox_small_int8.bin
```

If they have not been generated yet, run the model preparation and quantization scripts.

---

# Running FP16

```env
MODEL_PRECISION=FP16
```

The FP16 OpenVINO model is prepared from the YOLOX-S ONNX model if required.

---

# Running FP32

```env
MODEL_PRECISION=FP32
```

FP32 is useful as the baseline precision for comparison and benchmarking.

---

# Important Notes

## Single-worker FastAPI

Live job progress is currently stored in process memory.

For this reason, use a single Uvicorn worker:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

Using multiple workers would require moving job-progress state to a shared system such as Redis or a database.

---

## OpenRouter

The OpenRouter API key is optional.

Without it, cloud Gemma analysis is skipped.

Never commit API keys to GitHub.

---

## INT8 Quantization

INT8 is generated through post-training quantization using representative calibration frames.

Calibration data should ideally be separate from final benchmark/evaluation videos.

---

# Known Limitations

This project is currently intended as a research/demo pipeline rather than a production traffic-enforcement system.

Current limitations include:

- CPU-focused deployment
- in-memory live progress state
- single-worker web deployment
- PaddleOCR/model assets may require internet on first run
- local Gemma inference can be computationally expensive
- model accuracy depends strongly on camera angle, resolution, lighting, and traffic conditions
- heuristic vehicle attributes should not be treated as certified physical measurements
- INT8 performance and accuracy can vary depending on calibration data and hardware

---

# Security / Repository Hygiene

The project includes protections for:

- sanitized uploaded filenames
- allowed video extensions
- validated job IDs
- storage path containment
- isolated per-job directories
- automatic cleanup of old job data
- `.env` exclusion from Git
- generated weights/output exclusion where appropriate
- `.dockerignore` to prevent local secrets and unnecessary files from being copied into Docker images

---

# Future Improvements

Potential future work includes:

- further OpenVINO inference optimization
- improved model accuracy benchmarking
- automatic distribution/download of the prebuilt INT8 model
- more robust license-plate detection
- improved multi-object tracking
- shared job state for multi-worker deployment
- persistent job queues
- cloud/object-storage support
- stronger automated evaluation across FP32, FP16, and INT8

---

# License

The project code is licensed under the **MIT License**.

See:

```text
LICENSE
```

for details.

Third-party libraries, models, and model weights such as YOLOX, PaddleOCR, OpenVINO, Gemma, and LiteRT are subject to their respective licenses and terms.

---

# Author

**Bhuvanesh O**

GitHub: https://github.com/bhuvanesh-o

---

## Disclaimer

This project was developed for experimentation and research in traffic-video analytics, model optimization, multimodal AI, and ML-system deployment.

Outputs generated by OCR, object detection, heuristics, or multimodal language models may contain errors and should not be used as the sole basis for legal, safety-critical, or enforcement decisions.
