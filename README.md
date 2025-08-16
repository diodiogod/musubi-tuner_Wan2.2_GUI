# Musubi Tuner WAN 2.2 GUI

This repository is a fork of the original [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) project, enhanced with a graphical user interface (GUI) tailored for WAN 2.2 LoRA training. The GUI simplifies configuring, running, and monitoring LoRA training for WAN 2.2 models, making it accessible for users who prefer a visual interface over command-line operations.

The core functionality builds on the Musubi Tuner ecosystem, supporting Low-Rank Adaptation (LoRA) training for architectures like HunyuanVideo, WAN 2.1/2.2, FramePack, and FLUX.1 Kontext. This fork focuses on WAN 2.2, with features for handling high-noise and low-noise DiT models.

**Note**: This GUI is experimental and optimized for Windows (via the included .bat launcher), but it may work on other platforms with adjustments. Refer to the original repository for the full capabilities of Musubi Tuner.

## Features

The GUI provides an intuitive interface for setting up and executing WAN 2.2 LoRA training workflows. Key features include:

### Configuration Tabs
- **Model Paths & Dataset Tab**:
  - Specify the dataset configuration (TOML file) path.
  - Enable training for High Noise and/or Low Noise DiT models.
  - Set paths for VAE, optional CLIP, and T5 text encoder models.
  - Define the output directory and LoRA filename.
  - Includes file browsers for easy path selection.

- **Training Parameters Tab**:
  - Configure core parameters: learning rate, max epochs, save frequency, and seed.
  - Set LoRA network dimension (rank) and alpha.
  - Select optimizer type (e.g., `adamw8bit`, `prodigy`) and optional additional arguments.
  - Choose learning rate scheduler (e.g., `cosine`, `polynomial`) with power and min LR ratio options.

- **Advanced Settings Tab**:
  - Memory optimizations: Mixed precision (`fp16`/`bf16`), gradient checkpointing, persistent data loaders, gradient accumulation steps, and max data loader workers.
  - WAN 2.2-specific options: Offload inactive DiT model or swap blocks to save VRAM.
  - Flow matching: Timestep sampling (`uniform`/`shift`), discrete flow shift, and preserve distribution shape.
  - Attention mechanisms: Select from None, xFormers, Flash Attention, or SDPA via radio buttons.
  - Logging: Enable TensorBoard or Weights & Biases (W&B) with customizable log directory and prefix.

- **Run & Monitor Tab**:
  - Real-time console output for training progress. 
  - Progress bar showing current epoch and completion percentage.
  - Live loss graph (requires `matplotlib`; optional). (x axis range should be divided by batch size value but it's ok)*
  - VRAM usage monitor with current, peak, and total usage (requires `pynvml`; optional).
  - Buttons to start/stop training, show the generated command, and clear the console.

### Additional Functionality
- **Settings Management**: Load/save configurations from JSON files, reset to defaults, and auto-save last settings on close.
- **Validation**: Checks required fields before training; highlights invalid inputs.
- **Caching Support**: Optionally recache latents and text encoder outputs before training.
- **Command Building**: Automatically generates and executes commands for caching and training using `accelerate launch`.
- **Tooltips**: Hover over fields for explanations and tips.
- **FP8 Handling**: Automatically adjusts mixed precision for FP8 base models, with warnings.
- **Sequence Execution**: Runs caching and training steps sequentially, halting on errors.
- **Resume Training**: Supports resuming from checkpoints, from network weights, and advanced flags like FP8 scaled/base.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PGCRT/musubi-tuner_Wan2.2_GUI.git
   cd musubi-tuner_Wan2.2_GUI

Set Up Virtual Environment (Recommended):

Create and activate a Python 3.10+ virtual environment.

bashpython -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/macOS

Install Dependencies:

Install PyTorch (CUDA-enabled recommended):
bashpip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Install the Musubi Tuner package:
bashpip install -e .

Optional dependencies for full GUI features:
bashpip install matplotlib pynvml tensorboard wandb

matplotlib: For live loss graphing.
pynvml: For VRAM monitoring (NVIDIA GPUs only).

tensorboard/wandb: For logging (selected in GUI):
On Windows:
.\venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

wandb login

Then paste your API key




Model Downloads:

Follow the instructions in the original Musubi Tuner README for downloading WAN 2.2 models (DiT high/low noise, VAE, text encoders).
Place models in the directory structure as described in the original repository. 
SCALED MODELS won't work! Uses fp16 models with fp8 training, works on RTX 4090

Usage
Launching the GUI

Windows: Double-click LAUNCH_GUI.bat to activate the virtual environment and start the GUI.
Other Platforms: Run python musubi_tuner_gui.py from the activated virtual environment.

The GUI opens with the title "Musubi Tuner GUI - WAN 2.2 LoRA Training" (resizable, default size 1200x900).
Configuring Training

Fill in Paths and Parameters:

Use the tabs to enter required fields (e.g., dataset TOML, model paths).
Tooltips provide guidance for each field.
Use browse buttons to select files/directories.


Load/Save Settings:

Click "Load Settings" to import from a JSON file.
Click "Save Settings" to export the current configuration.
Click "Reset to Defaults" to load predefined safe values.


About Caching:

Enable caching for latents or text encoders if needed. (Check boxes on the first run, uncheck for the next one or if resume, rebake cache if adding images/videos to your dataset, or if you change resolution of the training)



Known Limitations

The GUI is optimized for WAN 2.2 T2V ONLY, You can use it to generate command line for I2V, then change the timesteps to the recommended ones for I2V, then run the command with the native script like on the official repo.


License
Licensed under the Apache License 2.0, consistent with the original Musubi Tuner. See the LICENSE file for details.
For additional details on the underlying Musubi Tuner, refer to the original repository.
