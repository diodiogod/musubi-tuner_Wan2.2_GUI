# Musubi Tuner WAN 2.2 GUI

A graphical user interface for WAN 2.2 LoRA training, built on top of the [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project. This GUI simplifies the configuration, execution, and monitoring of LoRA training workflows for WAN 2.2 models.

## Overview

This fork provides an intuitive visual interface for training LoRA adapters on WAN 2.2 models, supporting both high-noise and low-noise DiT architectures. The GUI handles complex parameter configurations, real-time monitoring.

**Note**: This GUI is experimental and optimized for Windows systems. Other platforms may require adjustments.

## Features

### 🎛️ Configuration Tabs

**Model Paths & Dataset**
- Dataset configuration (TOML file) selection
- High/Low Noise DiT model training toggles
- VAE, CLIP, and T5 text encoder path configuration
- Output directory and LoRA filename specification
- Built-in file browsers for easy navigation

**Training Parameters**
- Core settings: learning rate, epochs, save frequency, seed
- LoRA configuration: network dimension (rank) and alpha
- Optimizer selection (adamw8bit, prodigy, etc.)
- Learning rate scheduler options

**Advanced Settings**
- Memory optimizations: mixed precision, gradient checkpointing, data loader settings
- WAN 2.2 specific: DiT model offloading and block swapping for VRAM efficiency
- Flow matching: timestep sampling and distribution controls
- Attention mechanisms: xFormers, Flash Attention, SDPA support
- Logging integration: TensorBoard and Weights & Biases

**Run & Monitor**
- Real-time console output and progress tracking
- Live loss visualization (requires matplotlib)
- VRAM usage monitoring (requires pynvml)
- Training control: start, stop, and command preview

### 🔧 Additional Features

- **Settings Management**: Save/load configurations, auto-save on exit
- **Input Validation**: Required field checking with visual feedback
- **Command Generation**: View and copy generated training commands
- **Resume Training**: Checkpoint loading and advanced resumption options

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU (recommended for optimal performance)
- 12GB+ VRAM (24GB+ recommended for high-resolution training)
- 32GB+ system RAM

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI.git
   cd musubi-tuner_Wan2.2_GUI
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   
   # Install Musubi Tuner
   pip install -e .
   
   # Optional: Install GUI enhancement packages
   pip install matplotlib pynvml tensorboard

4. **Configure Weights & Biases (Optional)**
   ```bash
   pip install wandb
   ```
   ```bash
   wandb login
   # Enter your API key when prompted
   ```

5. **Download Models**
   2.2 HIGH NOISE https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors
   2.2 LOW NOISE https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors

   T5 https://huggingface.co/Kijai/WanVideo_comfy/blob/main/umt5-xxl-enc-bf16.safetensors

   VAE https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors

   **Important**: Use fp16 models only. Scaled models are not supported.

## Usage

### Launching the GUI

**Windows**: Double-click `LAUNCH_GUI.bat`

**Other Platforms**: 
```bash
python musubi_tuner_gui.py
```

### Basic Workflow

EDIT your .TOML dataset file in the dataset folder, IMPORTANT! Don't use backslashes "\" when you paste your dataset path, use "/" slashs instead (on notepad ++, CTRL+H to replace all backslashes)

Launch ACCELERATE config (last tab, follow instructions, only once)

1. **Configure Paths**
   - Set your dataset TOML file path
   - Specify model paths (DiT, VAE, text encoders)
   - Choose output directory and LoRA filename

2. **Set Training Parameters**
   - Configure learning rate, epochs, and network dimensions
   - Select optimizer and scheduler options
   - Adjust advanced settings as needed

3. **Enable Caching (First Run)**
   - Check latents and/or text encoder caching options
   - Required for initial training or when dataset changes

4. **Start Training**
   - Switch to "Run & Monitor" tab
   - Click "Start Training" to begin
   - Monitor progress via console, graph, and VRAM usage

5. **Save Configuration**
   - Use "Save Settings" to save your configuration
   - Settings auto-save on GUI close anyway

*   **IMPORTANT:** Use caching on your first run or when adding new data to your dataset to speed up initialization.
*   Monitor VRAM usage to optimize memory settings with `blocks_to_swap`, batch size, and resolution.


### Dual Model Training Methods
Here are the four distinct methods for training both the High and Low noise models.

---

### 1. Manually Sequential Training
*This is a fully manual workflow where you run two separate training jobs yourself, one for each model.*

*   **How to Configure:**
    1.  **Part 1:** Check **only** `Train High Noise Model` and run the training to completion.
    2.  **Part 2:** Return, uncheck the low noise model, check **only** `Train Low Noise Model`, and start the second training.

*   **Key Attributes:**
    *   **VRAM Usage:** **Moderate.** Very friendly to lower-VRAM systems.
    *   **Training Speed:** **Good.** Each session runs at full speed.
    *   **Best Use Case:** The **highly recommended alternative** to Method 2 for users with limited VRAM. It is predictable, efficient, and avoids the crippling speed loss.

 ---

### 2. Sequentially (Two Separate Trainings)
*This method automatically runs two training sessions back-to-back when you set different LoRA parameters for each model.*

*   **How to Configure:**
    1.  Go to the **"Model Paths & Dataset"** tab and check **both** `Train High Noise Model` and `Train Low Noise Model`.
    2.  Go to the **"Training Parameters"** tab and enter **different** values for `Network Dimension (Rank)` or `Network Alpha` for the high noise model.

*   **Key Attributes:**
    *   **VRAM Usage:** **Moderate.** Only one DiT model is loaded into VRAM at a time.
    *   **Training Speed:** **Good.** Each session runs at full speed.
    *   **Best Use Case:** Required when you intentionally want different LoRA ranks or alphas for each model, using the same dataset and learning rate. / If you want to automate training both models            during the night.

---

### 3. Combined (Single Run - Very High VRAM Usage)
*This is the fastest and most efficient method, running a single unified process that trains both models simultaneously by keeping them both in VRAM.*
Similar to AI toolkit when both models are selected with LOW VRAM "OFF"

*   **How to Configure:**
    1.  Go to the **"Model Paths & Dataset"** tab and check **both** `Train High Noise Model` and `Train Low Noise Model`.
    2.  In the **"Training Parameters"** tab, leave the high noise model's `Network Dimension` and `Network Alpha` **blank**.
    3.  In the **"Advanced Settings"** tab, ensure `Offload Inactive DiT Model` is **UNCHECKED**.

*   **Key Attributes:**
    *   **VRAM Usage:** **Very High.** Requires enough VRAM to hold both DiT models (Too much for consumer grade GPU).
    *   **Training Speed:** **Fastest.** The most time-efficient method.
    *   **Best Use Case:** The default and highly recommended method for users with sufficient VRAM.

---

### 4. Combined (Single Run - VRAM Saving Mode)
*This method attempts a combined run on low-VRAM systems by constantly swapping the inactive model between VRAM and system RAM, causing a high performance bottleneck.*
Similar to AI toolkit when both models are selected with LOW VRAM "ON"

*   **How to Configure:**
    1.  Go to the **"Model Paths & Dataset"** tab and check **both** `Train High Noise Model` and `Train Low Noise Model`.
    2.  In the **"Training Parameters"** tab, leave the high noise model's `Network Dimension` and `Network Alpha` **blank**.
    3.  In the **"Advanced Settings"** tab, **CHECK** the `Offload Inactive DiT Model` option.

*   **Key Attributes:**
    *   **VRAM Usage:** **Low.** Drastically reduces VRAM consumption.
    *   **Training Speed:** **Slow.** The performance cost is crippling.
    *   **Best Use Case:**
        > **WARNING:** This option has no real practical use case and should be avoided. The speed penalty is so significant that it is almost always slower than running two separate trainings (Method 1 or 2).

---



## License

Licensed under the Apache License 2.0, consistent with the original Musubi Tuner project. See the [LICENSE](LICENSE) file for full details.

## Acknowledgments

Built upon the excellent work of the [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project. Special thanks to the original contributors and the broader AI training community.

---

**Need Help?** Check the [Issues](https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI/issues) page or refer to the [original Musubi Tuner documentation](https://github.com/kohya-ss/musubi-tuner) for additional guidance.
