# LTX Video Generation Pipeline

This project utilizes **LTX-Video**, a deep-learning model for generating high-quality AI videos using the **Diffusers library**. The project is designed to run on **Windows (VS Code + Jupyter Notebook)** and supports GPU acceleration via **CUDA**.

## 🚀 Features
- **AI-Powered Video Generation**: Generate realistic AI-generated videos based on text prompts.
- **Customizable Parameters**: Adjust resolution, frame count, and inference steps for better quality.
- **GPU Support**: Leverages CUDA for accelerated processing.
- **Manual Model Download Support**: Option to manually download the 40GB model for faster access.

---

## 🔹 Minimum System Requirements
Since this project is computationally expensive, the **minimum laptop specs** required are:

### ✅ **Minimum Recommended Specs** *(Same as Developer's System)*
- **Processor**: Intel Core ultra 9 or AMD Ryzen 9 7945HX
- **RAM**: 32GB DDR5
- **GPU**: NVIDIA RTX 4070 (8GB VRAM) or higher
- **Storage**: At least 100GB free SSD space
- **CUDA Version**: 11.8 or higher

### 🔥 **Higher Performance Setup (Recommended for Faster Processing)**
- **Processor**: Intel Core i9-14900HX / AMD Ryzen 9 7950X
- **RAM**: 64GB DDR5
- **GPU**: NVIDIA RTX 4090 (16GB+ VRAM)
- **Storage**: NVMe SSD with 1TB free space
- **CUDA Version**: 12.1+

⚠ **Note**: The higher the number of inference steps, the heavier the processing power required. Expect longer generation times for **high-quality, long-duration videos**.

---

## 🛠 Installation Guide

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/mohdalipatel8976/ai-video-generator.git
cd ltx-video-generation
```

### **Step 2: Create a Virtual Environment (Recommended)**
```bash
python -m venv video_env
```

### **Step 3: Activate the Virtual Environment**
- **Windows**:
  ```bash
  video_env\Scripts\activate
  ```
- **Linux/Mac**:
  ```bash
  source video_env/bin/activate
  ```

### **Step 4: Install Dependencies**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install huggingface_hub numpy scipy opencv-python tqdm imageio imageio-ffmpeg
```

---

## 📥 **Downloading the LTX-Video Model (40GB)**
The **LTX-Video model is ~40GB**, and downloading it via `from_pretrained()` might be slow. Instead, follow these steps:

### **Method 1: Use Hugging Face CLI (Recommended)**
```bash
huggingface-cli download Lightricks/LTX-Video --local-dir "D:/huggingface_models/LTX-Video"
```

### **Method 2: Download Manually**
1. Visit **[Hugging Face: Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video)**.
2. Download the following files:
   - `model_index.json`
   - `diffusion_pytorch_model.safetensors`
   - `config.json`
   - `scheduler_config.json`
3. Save them inside a folder (e.g., `D:/huggingface_models/LTX-Video`).

### **Step 5: Modify Your Script to Load the Model Locally**
Update your Python script to use the manually downloaded model:
```python
from diffusers import LTXPipeline
pipe = LTXPipeline.from_pretrained("D:/huggingface_models/LTX-Video", torch_dtype=torch.bfloat16)
```

---

## 🎬 Running the Video Generation Script
1. Open **Jupyter Notebook** inside VS Code.
2. Run the provided `generate_video.ipynb` notebook.
3. Once the video is generated, it will be saved as **output.mp4** in the project folder.

```python
output_path = "output.mp4"
export_to_video(video, output_path, fps=24)
print(f"✅ Video saved successfully as {output_path}!")
```

### **Opening the Video in Windows Automatically**
```python
import os
os.startfile("output.mp4")
```

---

## 🛠 Troubleshooting
### **1️⃣ Stuck at Model Download (Fetching Files 0%)?**
- Use a **VPN** to bypass slow regional downloads.
- Download manually from Hugging Face (see Step 5).

### **2️⃣ CUDA Not Available?**
- Run the following command:
  ```python
  import torch
  print("CUDA available:", torch.cuda.is_available())
  ```
- If `False`, update your **GPU drivers** and install **CUDA 11.8+**.

---

## 📝 License
This project is open-source and available under the **MIT License**.

---

## 🙌 Contributing
Feel free to **fork** this repository, improve the code, and submit pull requests!

🔗 **GitHub Repository:** [Your Repo Link Here]

💬 **Contact:** If you have questions, open an issue or reach out on GitHub.

---

### ✅ **Enjoy AI Video Generation! 🚀**

