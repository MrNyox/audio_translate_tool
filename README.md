Friend asked if i could make a tool for translating videos into arabic, i tried, i failed horibly , this my attempt for you to laugh at, it barely functions,it's held by nothing excpet soulless vibe coded slop , there are hundreds of repositories that did it better than i could ever, but hey attempt is attempt.
anyways here is the ai slop README.md that i asked Qwen3.8 max to generate for me, if something doesn't work on your machine then what can i say except skill issue .
also you should have at least an rtx 3060 12gb model if you want to run it locally on your pc , otherwise it will run on your cpu but it will be very slow, or you can try running it on a cloud provider like google colab or aws.
i provided a google colab notebook that you can use to run the model on your own machine.
here is the boring tutorial part if you're still reading to this point. 

***

# 🎙️ Audio Translate Tool

## 📖 Description
The **Audio Translate Tool** is a locally hosted utility designed for dubbing and translating videos using Alibaba's Qwen language models. Built around a Flask web interface, this tool leverages state-of-the-art speech recognition (ASR) capabilities to process audio and video files, making it a powerful utility for translation and dubbing workflows.

## ⚙️ Requirements
- **Python**: Version 3.8 or higher.
- **System Dependencies**: `FFmpeg` (essential for video/audio processing).
- **Python Libraries** (from `requirements.txt`):
  - `Flask>=3.0.0` (Web Framework)
  - `torch>=2.3.0` (Deep Learning Framework)
  - `qwen-asr>=0.0.6` (Qwen Audio Speech Recognition integration)
  - `transformers>=4.44.0` & `accelerate>=0.29.0`
  - `soundfile>=0.12.1` & `sentencepiece>=0.2.0`
  - `huggingface_hub>=0.24.0`
  - `bitsandbytes>=0.43.0` & `fonttools>=4.50.0`

## 🤖 Models Used
- **Qwen-ASR (Qwen Audio Speech Recognition)**: The primary model used for transcribing and processing audio. It is an open-source model series provided by the Alibaba Qwen Team [[3]]. The `qwen-asr` PyPI package handles the integration and automatic fetching of the model weights via the Hugging Face Hub [[1]].

---

## 🚀 Setup & Installation Guides

### 🪟 Windows Setup Guide

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MrNyox/audio_translate_tool.git
   cd audio_translate_tool
   ```

2. **Create and activate a virtual environment** (Highly Recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Requirements**:
   First, install PyTorch with CUDA support if you have an NVIDIA GPU for faster processing:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Then, install the rest of the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models**:
   The `qwen-asr` package automatically downloads the necessary Qwen-ASR models upon the first run. If you prefer to manually pre-download them into the `./models` folder, you can use the Hugging Face CLI:
   ```bash
   huggingface-cli download Qwen/Qwen-Audio-Chat --local-dir ./models
   ```
   *(Note: Replace `Qwen/Qwen-Audio-Chat` with the specific model ID required by your version of `qwen-asr` if it differs).*

5. **Launch the Tool**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

### 🐧 Linux Setup Guide

1. **Install System Dependencies**:
   You will need `ffmpeg` for media processing and some build tools to compile the requirements:
   ```bash
   sudo apt update
   sudo apt install ffmpeg python3-pip python3-venv git -y
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/MrNyox/audio_translate_tool.git
   cd audio_translate_tool
   ```

3. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Requirements**:
   Install PyTorch with CUDA support (Recommended for Linux environments with a GPU):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download Models**:
   Models will be automatically fetched by the `qwen-asr` library when you first run the application. To manually cache or download the model to your local directory:
   ```bash
   huggingface-cli download Qwen/Qwen-Audio-Chat --local-dir ./models
   ```

6. **Launch the Tool**:
   ```bash
   python app.py
   ```
   The application will spin up a Flask server. Access the UI at `http://127.0.0.1:5000`.

---

## 🏃 How to Launch (After Downloading Requirements)

Once you have completed the setup steps above for your respective OS and the requirements are installed, launching the tool is straightforward:

1. Ensure your **virtual environment** is activated (`venv\Scripts\activate` on Windows or `source venv/bin/activate` on Linux).
2. Navigate to the root directory of the repository.
3. Run the main Flask application script:
   ```bash
   python app.py
   ```
4. The server will start in debug mode on `127.0.0.1` port `5000`. Open any web browser and go to:
   👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** 

You are now ready to start using the dubbing and translation tool!
