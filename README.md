# Musubi Tuner WAN 2.2 GUI

A graphical user interface for WAN 2.2 LoRA training, built on top of the [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project. This GUI simplifies the configuration, execution, and monitoring of LoRA training workflows for WAN 2.2 models.

## Overview

This fork provides an intuitive visual interface for training LoRA adapters on WAN 2.2 models, supporting both high-noise and low-noise DiT architectures. The GUI handles complex parameter configurations, real-time monitoring, and automated caching processes while maintaining compatibility with the underlying Musubi Tuner ecosystem.

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
- **Caching Support**: Automated latent and text encoder caching
- **Command Generation**: View and copy generated training commands
- **Resume Training**: Checkpoint loading and advanced resumption options
- **FP8 Compatibility**: Automatic precision adjustments with warnings

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
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   
   # Install Musubi Tuner
   pip install -e .
   
   # Optional: Install GUI enhancement packages, xformers
   pip install matplotlib pynvml tensorboard wandb
   pip install xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Configure Weights & Biases (Optional)**
   ```bash
   pip install wandb
   ```
   ```bash
   wandb login
   # Enter your API key when prompted
   ```

5. **Download Models**
   Follow the original [Musubi Tuner documentation](https://github.com/kohya-ss/musubi-tuner) to download WAN 2.2 models (DiT high/low noise, VAE, text encoders).

   **Important**: Use fp16 models only. Scaled models are not supported.

## Usage

### Launching the GUI

**Windows**: Double-click `LAUNCH_GUI.bat`

**Other Platforms**: 
```bash
python musubi_tuner_gui.py
```

### Basic Workflow

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
   - Use "Save Settings" to preserve your configuration
   - Settings auto-save on GUI close

### Tips

- Enable both high and low noise models for complete WAN 2.2 training (not for consumer grade GPU) (can check offload inactive DITs if both models selected, AUTOMATICALLY disable BLOCKS SWAPs)
- Use caching on first run or when adding new data to your dataset
- Monitor VRAM usage to optimize memory settings
- Save successful configurations for future use

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| VRAM | 12GB | 24GB+ |
| System RAM | 16GB | 32GB+ |
| GPU | NVIDIA RTX 3080 | RTX 4090+ |
| Storage | 50GB free | 100GB+ SSD |

## Known Limitations

- **I2V Training**: GUI optimized for Text-to-Video only. For Image-to-Video, generate commands via GUI and run with native scripts
- **Multi-GPU**: Single GPU training only
- **Platform**: Best performance on Windows; other platforms may need adjustments
- **Dependencies**: Some features require optional packages (matplotlib, pynvml)

## License

Licensed under the Apache License 2.0, consistent with the original Musubi Tuner project. See the [LICENSE](LICENSE) file for full details.

## Acknowledgments

Built upon the excellent work of the [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project. Special thanks to the original contributors and the broader AI training community.

---

**Need Help?** Check the [Issues](https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI/issues) page or refer to the [original Musubi Tuner documentation](https://github.com/kohya-ss/musubi-tuner) for additional guidance.
