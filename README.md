# ASL-ML: Real-Time American Sign Language Translator

![ASL Alphabet Demo](demo/Demo-alphabet.gif)
![ASL Dynamic Word Demo](demo/Demo-word.gif)

## Overview

ASL-ML is a real-time American Sign Language (ASL) translation system that converts hand gestures into text using computer vision, deep learning, and natural language processing. The system features two primary modes: Static Alphabet recognition for fingerspelling and Dynamic Word recognition for continuous gestures.

## Key Features

*   **Real-Time Hand Tracking:** Powered by MediaPipe for high-performance 21-point landmark extraction.
*   **Static Recognition:** PyTorch-based 29-class classifier (A-Z, space, delete, nothing) optimized via ONNX.
*   **Dynamic Recognition:** LSTM-based sequence processing for continuous gestures (e.g., "hello", "thank you").
*   **LLM Integration:** GPT-2 driven word suggestions for context-aware sentence completion.
*   **Virtual Cursor:** Two-handed gesture control system for UI interaction via pinch-to-grab logic.
*   **Stabilization Engine:** Multi-frame consistency checks to ensure prediction accuracy.

## System Architecture

```
                    Webcam Feed
                        |
                        v
              +-----------------+
              |  React Frontend |
              +-----------------+
                        |
              WebSocket (base64 frames)
                        |
                        v
              +-----------------+
              | FastAPI Backend |
              +-----------------+
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
    +----------+   +---------+   +----------+
    | MediaPipe|   | Letter  |   |  Word    |
    | Hand     |-->| Recognizer|-->| Predictor|
    | Tracker  |   | (ONNX)  |   | (GPT-2)  |
    +----------+   +---------+   +----------+
          |             |
          v             v
    +----------+   +---------+
    | Stabilizer|  | Dynamic   |
    | (frames)  |  | Recognizer|
    +----------+   +---------+
          |             |
          +------+------+
                 |
                 v
          Response (text, suggestions, video)
```

## Tech Stack

*   **Frontend:** React, Tailwind CSS, Lucide React
*   **Backend:** FastAPI, Python, WebSockets
*   **Machine Learning:** PyTorch, ONNX, MediaPipe, HuggingFace (GPT-2)
*   **DevOps:** Docker, Docker Compose

## Project Structure

```
ASL-ML/
├── backend/           # FastAPI server, tracking, and inference logic
├── frontend/          # React application and UI components
├── models/            # Pre-trained ONNX and NLP models
├── scripts/           # Training, data collection, and utility scripts
├── data/              # Processed landmarks and custom datasets
└── docker-compose.yml # Container orchestration
```

## Installation

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   Webcam

### Quick Start
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ASL-ML.git
    cd ASL-ML
    ```

2.  **Setup Backend:**
    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    python ../scripts/download_mediapipe_model.py
    ```

3.  **Setup Frontend:**
    ```bash
    cd ../frontend
    npm install
    ```

4.  **Run Application:**
    *   **Manual:** Start backend (`python main.py` in `backend/`) and frontend (`npm start` in `frontend/`).
    *   **Docker:** `docker-compose up --build`

## Model Implementation

### Static Model
*   **Architecture:** Multi-layer Perceptron (MLP) with 3 hidden layers.
*   **Input:** 63 normalized landmark coordinates.
*   **Optimization:** Exported to ONNX for low-latency inference.

### Dynamic Model
*   **Architecture:** 2-layer LSTM with 128 hidden units.
*   **Processing:** 30-frame sliding window sequences.
*   **Normalization:** Translation and scale-invariant preprocessing to ensure user-agnostic performance.

## Contributors

* @khodorrhajj
* @giokhadra

## License

This project is licensed under the MIT License.
