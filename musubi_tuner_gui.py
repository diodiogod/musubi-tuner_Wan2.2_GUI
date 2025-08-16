import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import json
import os
import re
import time
import sys
from pathlib import Path

# --- Dependency Check ---
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use("TkAgg")
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except Exception:
    PYNVML_AVAILABLE = False

# --- Helper Class for Tooltips ---
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        try:
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
        except Exception:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify='left',
                         background="#FFFFE0", relief='solid', borderwidth=1,
                         font=("Calibri", "10", "normal"), wraplength=400)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = None

# --- Main Application ---
class MusubiTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Musubi Tuner GUI - WAN 2.2 LoRA Training")
        self.root.geometry("1200x900")

        self.entries = {}
        self.hidden_frames = {}
        self.setup_styles()
        
        self.current_process = None
        self.monitoring_active = False
        self.vram_thread = None
        self.loss_data = []
        self.peak_vram = 0
        self.command_sequence = []
        self.fp8_warning_shown = False
        self.last_line_was_progress = False

        self.create_interface()
        self.load_default_settings()
        self._load_last_settings()
        self.update_button_states()

    def setup_styles(self):
        BG_COLOR = '#2B2B2B'; TEXT_COLOR = '#D3D3D3'; FIELD_BG_COLOR = '#3C3F41'
        SELECT_BG_COLOR = '#4A6185'; BORDER_COLOR = '#555555'; ERROR_BORDER = '#E53935'
        
        self.root.configure(bg=BG_COLOR)
        style = ttk.Style()
        try: style.theme_use('clam')
        except Exception: pass
        
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Calibri', 9))
        style.configure('TLabel', font=('Calibri', 10)); style.configure('TFrame', background=BG_COLOR)
        style.configure('TLabelframe', background=BG_COLOR, bordercolor=BORDER_COLOR, relief='solid', borderwidth=1)
        style.configure('TLabelframe.Label', background=BG_COLOR, foreground=TEXT_COLOR, font=('Calibri', 11, 'bold'))
        style.configure('TNotebook', background=BG_COLOR, borderwidth=0)
        style.configure('TNotebook.Tab', background='#3C3F41', foreground=TEXT_COLOR, padding=[10, 5], borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', BG_COLOR)])
        style.configure('TButton', background='#3C3F41', foreground=TEXT_COLOR, font=('Calibri', 10), borderwidth=1, relief='solid')
        style.map('TButton', background=[('active', '#4E5254'), ('pressed', '#585C5E')], bordercolor=[('active', BORDER_COLOR)], foreground=[('disabled', '#6A6A6A')])
        style.configure('TEntry', foreground=TEXT_COLOR, fieldbackground=FIELD_BG_COLOR, insertcolor=TEXT_COLOR, borderwidth=1, relief='solid', bordercolor=BORDER_COLOR, padding=3)
        style.map('TCombobox', fieldbackground=[('readonly', FIELD_BG_COLOR)], foreground=[('readonly', TEXT_COLOR)], selectbackground=[('readonly', SELECT_BG_COLOR)])
        self.root.option_add('*TCombobox*Listbox.background', FIELD_BG_COLOR); self.root.option_add('*TCombobox*Listbox.foreground', TEXT_COLOR)
        self.root.option_add('*TCombobox*Listbox.selectBackground', SELECT_BG_COLOR); self.root.option_add('*TCombobox*Listbox.selectForeground', TEXT_COLOR)
        style.configure('TCheckbutton', font=('Calibri', 10)); style.configure('Title.TLabel', font=('Calibri', 16, 'bold'))
        style.configure('Status.TLabel', font=('Calibri', 11, 'bold')); style.configure('TProgressbar', thickness=20, background=SELECT_BG_COLOR, troughcolor=FIELD_BG_COLOR)
        style.configure('Invalid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=ERROR_BORDER, foreground=TEXT_COLOR, relief='solid', borderwidth=1)
        style.configure('Valid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=BORDER_COLOR, foreground=TEXT_COLOR, relief='solid', borderwidth=1)

    def create_interface(self):
        self.root.grid_columnconfigure(0, weight=1); self.root.grid_rowconfigure(0, weight=1)
        canvas = tk.Canvas(self.root, bg='#2B2B2B', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", tags="frame")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig('frame', width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew"); scrollbar.grid(row=0, column=1, sticky="ns")
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        main_frame = ttk.Frame(scrollable_frame); main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(main_frame, text="Musubi Tuner - WAN 2.2 LoRA Training", style='Title.TLabel').pack(pady=(0, 20), anchor='w')
        self.create_settings_buttons(main_frame)
        
        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill="both", expand=True, pady=(10, 0))
        
        self.create_model_paths_tab()
        self.create_training_params_tab()
        self.create_advanced_tab()
        self.create_run_monitor_tab()
        self.create_convert_lora_tab()
        self.create_accelerate_config_tab() # --- ADDED ---

    def create_settings_buttons(self, parent):
        button_frame = ttk.Frame(parent); button_frame.pack(fill="x", pady=(0, 10), anchor='w')
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.load_default_settings).pack(side="left", padx=5)

    def _add_widget(self, parent, key, label, tooltip, kind='entry', options=None, is_required=False, validate_num=False, is_path=False, is_dir=False, default_val=False, command=None):
        frame = ttk.Frame(parent); frame.pack(fill="x", padx=5, pady=(5, 8))
        if kind != 'checkbox': ttk.Label(frame, text=label).pack(anchor="w")
        
        widget = None
        if kind == 'path_entry':
            path_frame = ttk.Frame(frame); path_frame.pack(fill="x", pady=(2, 0))
            widget = ttk.Entry(path_frame)
            widget.pack(side="left", fill="x", expand=True)
            filetypes = options if isinstance(options, list) else None
            def browse():
                path = filedialog.askdirectory() if is_dir else filedialog.askopenfilename(filetypes=filetypes)
                if path: widget.delete(0, tk.END); widget.insert(0, path); self.update_button_states()
            ttk.Button(path_frame, text="Browse", command=browse).pack(side="right", padx=(5, 0))
        elif kind == 'combobox':
            widget = ttk.Combobox(frame, values=options, state="readonly")
            if options: widget.set(options[0])
            widget.pack(fill="x", pady=(2, 0)); widget.bind("<MouseWheel>", lambda e: "break")
            if command: widget.bind("<<ComboboxSelected>>", command)
        elif kind == 'checkbox':
            var = tk.BooleanVar(value=default_val)
            def chained_command(event=None):
                if command and callable(command): command()
                self.update_button_states()
            widget = ttk.Checkbutton(frame, text=label, variable=var, command=chained_command)
            widget.var = var; widget.pack(anchor="w", padx=5, pady=2)
        else:
            vcmd = (self.root.register(self.validate_number), '%P') if validate_num else None
            widget = ttk.Entry(frame, validate="key", validatecommand=vcmd); widget.pack(fill="x", pady=(2, 0))

        if tooltip: ToolTip(widget, tooltip)
        self.entries[key] = widget
        widget.is_required = is_required; widget.is_path = is_path
        if isinstance(widget, ttk.Entry):
            widget.bind("<FocusOut>", self.update_button_states); widget.bind("<KeyRelease>", self.update_button_states)
        return widget
    
    def create_model_paths_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Model Paths & Dataset")
        
        dataset_frame = ttk.LabelFrame(frame, text="Dataset Configuration"); dataset_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(dataset_frame, "dataset_config", "Dataset Config (TOML):", "Path to .toml dataset configuration file.", kind='path_entry', options=[("TOML files", "*.toml")], is_required=True, is_path=True)
        
        dit_frame = ttk.LabelFrame(frame, text="DiT Model Selection"); dit_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(dit_frame, "is_i2v", "Is I2V Training?", "Enables Image-to-Video training mode. This changes some default behaviors and adds the --i2v flag.", kind='checkbox', command=self.update_button_states)
        
        high_noise_frame = ttk.LabelFrame(dit_frame, text="High Noise Model (T2V: 875-1000 / I2V: 900-1000)"); high_noise_frame.pack(fill="x", padx=5, pady=5)
        self._add_widget(high_noise_frame, "train_high_noise", "Train High Noise Model", "Enable to train the high noise model.", kind='checkbox', command=self.update_button_states)
        self._add_widget(high_noise_frame, "dit_high_noise", "DiT High Noise Model Path:", "Path to the high noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(high_noise_frame, "min_timestep_high", "Min Timestep:", "Minimum timestep for this model. (e.g., 875)", validate_num=True)
        self._add_widget(high_noise_frame, "max_timestep_high", "Max Timestep:", "Maximum timestep for this model. (e.g., 1000)", validate_num=True)

        low_noise_frame = ttk.LabelFrame(dit_frame, text="Low Noise Model (T2V: 0-875 / I2V: 0-900)"); low_noise_frame.pack(fill="x", padx=5, pady=(5,10))
        self._add_widget(low_noise_frame, "train_low_noise", "Train Low Noise Model", "Enable to train the low noise model.", kind='checkbox', command=self.update_button_states)
        self._add_widget(low_noise_frame, "dit_low_noise", "DiT Low Noise Model Path:", "Path to the low noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(low_noise_frame, "min_timestep_low", "Min Timestep:", "Minimum timestep for this model. (e.g., 0)", validate_num=True)
        self._add_widget(low_noise_frame, "max_timestep_low", "Max Timestep:", "Maximum timestep for this model. (e.g., 875)", validate_num=True)

        models_frame = ttk.LabelFrame(frame, text="Other Model Paths"); models_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(models_frame, "vae_model", "VAE Model:", "Path to VAE model (.safetensors or .pt). Required for training and caching.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(models_frame, "clip_model", "CLIP Model (Optional):", "Path to optional CLIP model. Required for I2V training.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(models_frame, "t5_model", "T5 Text Encoder:", "Path to T5 text encoder model. Required.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        
        output_frame = ttk.LabelFrame(frame, text="Output Configuration"); output_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(output_frame, "output_dir", "Output Directory:", "Base directory to save trained LoRAs. A subfolder will be automatically created.", kind='path_entry', is_dir=True, is_required=True, is_path=True)
        self._add_widget(output_frame, "output_name", "Output Name:", "Base filename for output LoRA (e.g., 'my_character'). Suffixes like '_LowNoise' will be added automatically.", is_required=True)

    def create_training_params_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Training Parameters")
        basic_frame = ttk.LabelFrame(frame, text="Basic Training Parameters"); basic_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(basic_frame, "learning_rate", "Learning Rate:", "The speed at which the model learns. Common values are 1e-4, 2e-4, 3e-4.", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "max_train_epochs", "Max Train Epochs:", "The total number of times the training process will iterate over the entire dataset.", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "save_every_n_epochs", "Save Every N Epochs:", "Frequency of saving checkpoints based on epochs. '1' saves after every epoch.", validate_num=True)
        self._add_widget(basic_frame, "save_every_n_steps", "Save Every N Steps:", "Frequency of saving checkpoints based on steps. Leave empty to disable.", validate_num=True)
        self._add_widget(basic_frame, "seed", "Seed:", "A number to ensure reproducible training results. Any integer will do.", validate_num=True)
        
        network_container = ttk.Frame(frame); network_container.pack(fill="x", padx=10, pady=10)
        self.hidden_frames['low_noise_lora_params'] = ttk.LabelFrame(network_container, text="Low Noise LoRA Parameters")
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_dim_low", "Network Dimension (Rank):", "The 'size' or capacity of the LoRA. Higher values can capture more detail but may overfit. Common values: 32, 64, 128.", is_required=True, validate_num=True)
        self._add_widget(self.hidden_frames['low_noise_lora_params'], "network_alpha_low", "Network Alpha:", "Acts as a learning rate scaler for the LoRA weights. Often set to half of the Network Dimension.", is_required=True, validate_num=True)

        self.hidden_frames['high_noise_lora_params'] = ttk.LabelFrame(network_container, text="High Noise LoRA Parameters")
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_dim_high", "Network Dimension (Rank):", "Leave blank to use the same as the Low Noise model. If different, a separate training run will be executed.", is_required=False, validate_num=True)
        self._add_widget(self.hidden_frames['high_noise_lora_params'], "network_alpha_high", "Network Alpha:", "Leave blank to use the same as the Low Noise model.", is_required=False, validate_num=True)

        optimizer_frame = ttk.LabelFrame(frame, text="Optimizer Settings"); optimizer_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(optimizer_frame, "optimizer_type", "Optimizer Type:", "'adamw8bit' is a memory-efficient and stable default. 'prodigy' can also work well.", kind='combobox', options=["adamw", "adamw8bit", "adafactor", "lion", "prodigy"])
        self._add_widget(optimizer_frame, "optimizer_args", "Optimizer Args:", "Additional arguments for the optimizer, e.g., 'weight_decay=0.1'. Can be left blank.", kind='entry')
        
        lr_frame = ttk.LabelFrame(frame, text="Learning Rate Scheduler"); lr_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(lr_frame, "lr_scheduler", "LR Scheduler:", "Algorithm to adjust learning rate during training. 'cosine' is a reliable choice.", kind='combobox', options=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"], command=self.update_button_states)
        self.hidden_frames['lr_warmup'] = ttk.Frame(lr_frame)
        self._add_widget(self.hidden_frames['lr_warmup'], "lr_warmup_steps", "Warmup Steps:", "Number of initial steps where the learning rate gradually increases. Can be a fixed number or a ratio (e.g., 0.1 for 10% of total steps).", validate_num=True)
        self.hidden_frames['lr_restarts'] = ttk.Frame(lr_frame)
        self._add_widget(self.hidden_frames['lr_restarts'], "lr_scheduler_num_cycles", "Restart Cycles:", "Number of times the learning rate will be reset for the 'cosine_with_restarts' scheduler.", validate_num=True)
        self._add_widget(lr_frame, "lr_scheduler_power", "LR Scheduler Power:", "The exponent for the polynomial decay. Only used by the 'polynomial' scheduler.", validate_num=True)
        self._add_widget(lr_frame, "lr_scheduler_min_lr_ratio", "Min LR Ratio:", "The minimum learning rate as a ratio of the initial learning rate.", validate_num=True)

    def create_advanced_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Advanced Settings")
        memory_frame = ttk.LabelFrame(frame, text="Memory & Performance"); memory_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(memory_frame, "mixed_precision", "Mixed Precision:", "Use 'fp16' or 'bf16' to reduce VRAM usage and speed up training. 'fp16' is common, 'bf16' is better on newer GPUs.", kind='combobox', options=["no", "fp16", "bf16"])
        self._add_widget(memory_frame, "gradient_checkpointing", "Gradient Checkpointing", "Drastically reduces VRAM usage by re-calculating gradients on the backward pass. Highly recommended.", kind='checkbox', default_val=True)
        self._add_widget(memory_frame, "persistent_data_loader_workers", "Persistent Data Loader Workers", "Keeps data loader processes alive between epochs to speed up data loading, at the cost of slightly higher RAM usage.", kind='checkbox')
        self._add_widget(memory_frame, "gradient_accumulation_steps", "Gradient Accumulation Steps:", "Simulates a larger batch size by accumulating gradients over several steps. E.g., a batch size of 1 with 4 accumulation steps simulates a batch size of 4.", validate_num=True)
        self._add_widget(memory_frame, "max_data_loader_n_workers", "Max Data Loader Workers:", "Number of CPU threads to load data. '2' is a safe default. Higher values can speed up loading but use more RAM.", validate_num=True)
        self._add_widget(memory_frame, "offload_inactive_dit", "Offload Inactive DiT Model", "When training both models in a combined run, offloads the inactive DiT model to CPU to save VRAM. Disables 'Blocks to Swap'.", kind='checkbox', command=self.update_button_states)
        self._add_widget(memory_frame, "blocks_to_swap", "Blocks to Swap:", "Number of DiT blocks to offload to CPU memory to save VRAM. Can slow down training. (e.g., 10)", validate_num=True)
        
        flow_frame = ttk.LabelFrame(frame, text="Flow Matching Parameters"); flow_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(flow_frame, "timestep_sampling", "Timestep Sampling:", "Method for selecting timesteps during training. 'shift' is recommended.", kind='combobox', options=["uniform", "shift", "sigma", "logsnr", "qinglong_flux"])
        self._add_widget(flow_frame, "num_timestep_buckets", "Timestep Buckets:", "Enables stratified sampling by dividing timesteps into buckets. Can improve training stability, especially with small datasets. (e.g., 10)", validate_num=True)
        self.hidden_frames['timestep_boundary'] = ttk.Frame(flow_frame)
        self._add_widget(self.hidden_frames['timestep_boundary'], "timestep_boundary", "Timestep Boundary:", "The point (as a ratio) where the model switches from high to low noise. Only used in combined training runs.", validate_num=True)
        self._add_widget(flow_frame, "discrete_flow_shift", "Discrete Flow Shift:", "Shift value for 'shift' sampling. The documentation recommends 3.0.", validate_num=True)
        self._add_widget(flow_frame, "preserve_distribution_shape", "Preserve Distribution Shape", "Prevents distortion of the timestep distribution. Recommended when training only one model (e.g., only low noise).", kind='checkbox')
        
        attention_frame = ttk.LabelFrame(frame, text="Attention Mechanism"); attention_frame.pack(fill="x", padx=10, pady=10)
        self.attention_var = tk.StringVar(value="xformers")
        self.entries['attention_mechanism'] = self.attention_var
        attention_options = [("None", "none"), ("xFormers", "xformers"), ("Flash Attention", "flash_attn"), ("SDPA", "sdpa")]
        for text, value in attention_options:
            rb = ttk.Radiobutton(attention_frame, text=text, variable=self.attention_var, value=value)
            rb.pack(anchor="w", padx=5, pady=2); ToolTip(rb, f"Optimized attention mechanism to save VRAM and increase speed. xFormers or Flash Attention are recommended if available.")

        logging_frame = ttk.LabelFrame(frame, text="Logging (TensorBoard / W&B)"); logging_frame.pack(fill="x", padx=10, pady=10)
        log_with_widget = self._add_widget(logging_frame, "log_with", "Log With:", "Enable logging with TensorBoard or Weights & Biases to monitor training progress.", kind='combobox', options=["none", "tensorboard", "wandb", "all"])
        log_with_widget.bind('<<ComboboxSelected>>', self.update_button_states)
        self._add_widget(logging_frame, "logging_dir", "Logging Directory:", "Directory to save logs. Required if 'Log With' is not 'none'.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(logging_frame, "log_prefix", "Log Prefix:", "Optional prefix for log filenames or wandb run names.", kind='entry')

        other_frame = ttk.LabelFrame(frame, text="Other Options"); other_frame.pack(fill="x", padx=10, pady=10)
        fp8_frame = ttk.Frame(other_frame); fp8_frame.pack(fill='x')
        self._add_widget(fp8_frame, "fp8_base", "FP8 Base", "Use FP8 precision for the base model. Requires bf16.", kind='checkbox', command=self._handle_fp8_precision_conflict)
        self._add_widget(fp8_frame, "fp8_scaled", "FP8 Scaled", "Use scaled FP8 training.", kind='checkbox')
        self._add_widget(fp8_frame, "fp8_t5", "FP8 T5", "Use FP8 precision for the T5 text encoder.", kind='checkbox')
        self._add_widget(other_frame, "save_state", "Save State", "Save the complete training state (optimizer, etc.) to allow resuming later.", kind='checkbox', default_val=True)
        
        resume_frame = ttk.LabelFrame(frame, text="Resume Training"); resume_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(resume_frame, "resume_path", "Resume from State:", "Path to a saved state folder to continue a previous training run.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(resume_frame, "network_weights", "Network Weights:", "Load pre-trained LoRA weights to continue training from them (fine-tuning a LoRA).", kind='path_entry', options=[("Weight files", "*.safetensors")], is_path=True)

    def create_run_monitor_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Run & Monitor")
        top_pane = ttk.Frame(tab_frame); top_pane.pack(fill='x', padx=10, pady=10)
        controls_frame = ttk.LabelFrame(top_pane, text="Controls & Caching"); controls_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        self.run_status_var = tk.StringVar(value="⚪ New Training RUN")
        self.run_status_label = ttk.Label(controls_frame, textvariable=self.run_status_var, style='Status.TLabel')
        self.run_status_label.pack(pady=5, padx=10)
        cache_opts_frame = ttk.Frame(controls_frame)
        cache_opts_frame.pack(pady=5, padx=10, fill='x')
        self._add_widget(cache_opts_frame, "recache_latents", "Re-cache Latents Before Training", "If your dataset or VAE changes, check this to force regeneration of the latent cache.", kind='checkbox')
        self._add_widget(cache_opts_frame, "recache_text", "Re-cache Text Encoders Before Training", "If your dataset or T5 model changes, check this to force regeneration of the text encoder cache.", kind='checkbox')
        train_button_frame = ttk.Frame(controls_frame); train_button_frame.pack(pady=10, padx=10, fill='x')
        self.start_btn = ttk.Button(train_button_frame, text="Start Training", command=self.start_training); self.start_btn.pack(side="left", padx=(0, 5), expand=True, fill='x')
        self.stop_btn = ttk.Button(train_button_frame, text="Stop Training", command=self.stop_training, state="disabled"); self.stop_btn.pack(side="left", padx=5, expand=True, fill='x')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, style='TProgressbar'); self.progress_bar.pack(pady=(5, 5), padx=10, fill='x')
        self.progress_label_var = tk.StringVar(value="Ready"); ttk.Label(controls_frame, textvariable=self.progress_label_var, anchor='center').pack(fill='x')
        monitor_frame = ttk.LabelFrame(top_pane, text="Live Monitoring"); monitor_frame.pack(side='left', fill='both', expand=True)
        self.vram_label_var = tk.StringVar(value="VRAM: N/A"); ttk.Label(monitor_frame, textvariable=self.vram_label_var).pack(anchor='w', padx=10, pady=5)
        self.peak_vram_label_var = tk.StringVar(value="Peak VRAM: N/A"); ttk.Label(monitor_frame, textvariable=self.peak_vram_label_var).pack(anchor='w', padx=10)
        ttk.Button(monitor_frame, text="Generate Command", command=self.show_command).pack(pady=(10,5), padx=10, fill='x')
        bottom_pane = ttk.PanedWindow(tab_frame, orient=tk.HORIZONTAL); bottom_pane.pack(fill='both', expand=True, padx=10, pady=10)
        graph_frame = ttk.LabelFrame(bottom_pane, text="Live Loss"); bottom_pane.add(graph_frame, weight=1)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(5, 4), dpi=100); self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.setup_graph_style()
        else: ttk.Label(graph_frame, text="Matplotlib not found.\nInstall with 'pip install matplotlib'", wraplength=200, justify='center').pack(expand=True)
        console_frame = ttk.LabelFrame(bottom_pane, text="Console Output"); bottom_pane.add(console_frame, weight=1)
        self.output_text = tk.Text(console_frame, wrap=tk.WORD, bg='#3C3F41', fg='#D3D3D3', insertbackground='#D3D3D3', font=('Consolas', 9), relief=tk.FLAT, bd=0)
        output_scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scrollbar.set); self.output_text.pack(side="left", fill="both", expand=True); output_scrollbar.pack(side="right", fill="y")

    def create_convert_lora_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Convert LoRA")
        main_frame = ttk.Frame(tab_frame); main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        settings_frame = ttk.LabelFrame(main_frame, text="Conversion Settings"); settings_frame.pack(fill='x', pady=(0,10))
        self._add_widget(settings_frame, "convert_lora_path", "LoRA to Convert:", "Path to the .safetensors LoRA file you want to convert.", kind='path_entry', options=[("Safetensors", "*.safetensors")], is_path=True)
        self._add_widget(settings_frame, "convert_output_path", "Output Path:", "Path to save the converted LoRA file.", kind='path_entry', options=[("Safetensors", "*.safetensors")])
        self._add_widget(settings_frame, "convert_precision", "Precision:", "The data type to save the converted LoRA in. 'fp16' is common for distribution.", kind='combobox', options=["fp16", "bf16", "float"])
        
        button = ttk.Button(settings_frame, text="Start Conversion", command=self.start_conversion); button.pack(pady=10)

        console_frame = ttk.LabelFrame(main_frame, text="Conversion Output"); console_frame.pack(fill='both', expand=True)
        self.convert_output_text = tk.Text(console_frame, wrap=tk.WORD, bg='#3C3F41', fg='#D3D3D3', insertbackground='#D3D3D3', font=('Consolas', 9), relief=tk.FLAT, bd=0)
        scrollbar = ttk.Scrollbar(console_frame, orient="vertical", command=self.convert_output_text.yview)
        self.convert_output_text.configure(yscrollcommand=scrollbar.set); self.convert_output_text.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        
    def create_accelerate_config_tab(self):
        # --- ADDED --- New tab for Accelerate config
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Accelerate Config")
        main_frame = ttk.Frame(tab_frame); main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        info_frame = ttk.LabelFrame(main_frame, text="Setup Instructions"); info_frame.pack(fill='x', pady=(0, 10))
        info_text_content = """This needs to be done only once before your first training run.
Click the button below to open a new terminal where you will configure Accelerate. Answer the questions based on your environment. For a standard single GPU setup, use the following answers:

- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only...?: NO
- Do you wish to optimize your script with torch dynamo?: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training...?: all
- Would you like to enable numa efficiency...?: NO
- Do you wish to use mixed precision?: bf16 (or fp16)

Note: If you get a 'ValueError: fp16 mixed precision requires a GPU', try answering '0' to the GPU question to explicitly select your first GPU.
"""
        info_text = tk.Text(info_frame, wrap=tk.WORD, bg='#3C3F41', fg='#D3D3D3', font=('Calibri', 10), relief=tk.FLAT, bd=0, height=15)
        info_text.insert(tk.END, info_text_content); info_text.config(state="disabled")
        info_text.pack(fill='x', expand=True, padx=10, pady=10)

        action_frame = ttk.LabelFrame(main_frame, text="Run Configuration"); action_frame.pack(fill='x')
        button = ttk.Button(action_frame, text="Run Accelerate Config", command=self.run_accelerate_config)
        button.pack(pady=20)

    def setup_graph_style(self):
        self.fig.patch.set_facecolor('#2B2B2B'); self.ax.set_facecolor('#3C3F41')
        self.ax.tick_params(axis='x', colors='white'); self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white'); self.ax.spines['top'].set_color('white') 
        self.ax.spines['right'].set_color('white'); self.ax.spines['left'].set_color('white')
        self.ax.yaxis.label.set_color('white'); self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white'); self.ax.set_xlabel("Steps"); self.ax.set_ylabel("Loss")
        self.canvas.draw()
    
    def validate_number(self, value):
        if value in ("", ".", "-"): return True
        try: float(value); return True
        except ValueError: return False

    def update_button_states(self, event=None):
        try:
            self._update_dynamic_widgets()
            if self.entries["resume_path"].get(): self.run_status_var.set("🟢 Resuming Training RUN")
            else: self.run_status_var.set("⚪ New Training RUN")
        except (KeyError, AttributeError): pass 

        all_valid = True
        train_high = self.entries["train_high_noise"].var.get(); train_low = self.entries["train_low_noise"].var.get()
        self.entries["dit_high_noise"].is_required = train_high; self.entries["dit_low_noise"].is_required = train_low
        self.entries["network_dim_low"].is_required = train_low; self.entries["network_alpha_low"].is_required = train_low
        self.entries["clip_model"].is_required = self.entries["is_i2v"].var.get()

        log_with = self.entries["log_with"].get(); self.entries["logging_dir"].is_required = log_with != "none"

        for widget in self.entries.values():
            if not isinstance(widget, tk.Widget): continue
            is_visible = False
            try:
                if widget.winfo_manager(): is_visible = True
            except tk.TclError: is_visible = False
            if not is_visible:
                if isinstance(widget, ttk.Entry): widget.config(style="Valid.TEntry")
                continue
            if isinstance(widget, ttk.Entry):
                is_valid = True
                if getattr(widget, 'is_required', False):
                    value = widget.get()
                    if not value: is_valid = False
                    elif getattr(widget, 'is_path', False) and not os.path.exists(value): is_valid = False
                style = "Valid.TEntry" if is_valid else "Invalid.TEntry"
                widget.config(style=style)
                if not is_valid: all_valid = False
        
        if not (train_high or train_low): all_valid = False
        self.start_btn.config(state="normal" if all_valid else "disabled")
        try:
            self.entries["recache_latents"].config(state="normal" if all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "vae_model"]) else "disabled")
            self.entries["recache_text"].config(state="normal" if all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "t5_model"]) else "disabled")
        except (AttributeError, KeyError): pass

    def _update_dynamic_widgets(self):
        show_low = self.entries["train_low_noise"].var.get(); show_high = self.entries["train_high_noise"].var.get()
        is_i2v = self.entries["is_i2v"].var.get()

        if show_low: self.hidden_frames['low_noise_lora_params'].pack(fill='x', expand=True, pady=(0, 5))
        else: self.hidden_frames['low_noise_lora_params'].pack_forget()
        if show_high: self.hidden_frames['high_noise_lora_params'].pack(fill='x', expand=True, pady=(0, 5))
        else: self.hidden_frames['high_noise_lora_params'].pack_forget()

        is_combined_run = show_low and show_high and not (self.entries["network_dim_high"].get() or self.entries["network_alpha_high"].get())

        if is_combined_run:
            self.hidden_frames['timestep_boundary'].pack(fill='x', expand=True)
            boundary_widget = self.entries["timestep_boundary"]
            current_val = boundary_widget.get()
            default_val = "0.9" if is_i2v else "0.875"
            if current_val != default_val: boundary_widget.delete(0, tk.END); boundary_widget.insert(0, default_val)
        else: self.hidden_frames['timestep_boundary'].pack_forget()

        offload_widget = self.entries["offload_inactive_dit"]; blocks_to_swap_widget = self.entries["blocks_to_swap"]
        offload_widget.config(state="normal" if is_combined_run else "disabled")
        if not is_combined_run: offload_widget.var.set(False) 
        is_offloading = offload_widget.var.get() and is_combined_run
        blocks_to_swap_widget.config(state="disabled" if is_offloading else "normal")
        if is_offloading: blocks_to_swap_widget.delete(0, tk.END)

        scheduler = self.entries["lr_scheduler"].get()
        if scheduler == "constant_with_warmup": self.hidden_frames['lr_warmup'].pack(fill='x', expand=True)
        else: self.hidden_frames['lr_warmup'].pack_forget()
        if scheduler == "cosine_with_restarts": self.hidden_frames['lr_restarts'].pack(fill='x', expand=True)
        else: self.hidden_frames['lr_restarts'].pack_forget()

    def get_settings(self):
        settings = {};
        for key, widget in self.entries.items():
            if isinstance(widget, (tk.BooleanVar, tk.StringVar)): settings[key] = widget.get()
            elif hasattr(widget, 'var'): settings[key] = widget.var.get()
            else: settings[key] = widget.get()
        return settings
    
    def set_values(self, settings):
        for key, value in settings.items():
            if key in self.entries:
                widget = self.entries[key]
                if isinstance(widget, (tk.BooleanVar, tk.StringVar)): widget.set(value)
                elif hasattr(widget, 'var'): widget.var.set(value)
                else: widget.delete(0, tk.END); widget.insert(0, str(value))
        self.update_button_states()

    def load_default_settings(self):
        defaults = {
            "dataset_config": "", "dit_high_noise": "", "dit_low_noise": "", "is_i2v": False,
            "train_high_noise": True, "train_low_noise": True,
            "min_timestep_low": "0", "max_timestep_low": "875", "min_timestep_high": "875", "max_timestep_high": "1000",
            "vae_model": "", "clip_model": "", "t5_model": "",
            "output_dir": "", "output_name": "my-lora",
            "learning_rate": "2e-4", "max_train_epochs": "10", "save_every_n_epochs": "1", "save_every_n_steps": "", "seed": "42",
            "network_dim_low": "32", "network_alpha_low": "16", "network_dim_high": "", "network_alpha_high": "",
            "optimizer_type": "adamw8bit", "optimizer_args": "", "lr_scheduler": "cosine",
            "lr_warmup_steps": "0", "lr_scheduler_num_cycles": "1",
            "mixed_precision": "fp16", "gradient_accumulation_steps": "1",
            "max_data_loader_n_workers": "2", "blocks_to_swap": "10", "timestep_sampling": "shift",
            "num_timestep_buckets": "", "timestep_boundary": "0.875", "discrete_flow_shift": "3.0", "preserve_distribution_shape": False,
            "gradient_checkpointing": True, "persistent_data_loader_workers": True, "save_state": True, 
            "fp8_base": False, "fp8_scaled": False, "fp8_t5": False, "offload_inactive_dit": False,
            "attention_mechanism": "xformers", "resume_path": "", "network_weights": "",
            "log_with": "none", "logging_dir": "", "log_prefix": "",
            "recache_latents": False, "recache_text": False,
            "convert_lora_path": "", "convert_output_path": "", "convert_precision": "fp16"
        }
        self.set_values(defaults)
        
    def _save_settings_to_file(self, filepath):
        try:
            with open(filepath, "w") as f: json.dump(self.get_settings(), f, indent=4); return True
        except Exception as e: print(f"Error saving settings to {filepath}: {e}"); return False

    def save_settings(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path and self._save_settings_to_file(file_path): messagebox.showinfo("Success", "Settings saved successfully!")

    def load_settings(self, filepath=None):
        if filepath is None: filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r") as f: settings = json.load(f)
                self.set_values(settings)
                if not filepath.endswith("last_settings.json"): messagebox.showinfo("Success", "Settings loaded successfully!")
            except Exception as e: messagebox.showerror("Error", f"Failed to load settings: {e}")

    def _load_last_settings(self): self.load_settings(filepath="last_settings.json")
    
    def _handle_fp8_precision_conflict(self):
        if self.entries["fp8_base"].var.get():
            precision_widget = self.entries["mixed_precision"]
            if precision_widget.get() != 'bf16':
                precision_widget.set('bf16')
                if not self.fp8_warning_shown:
                    messagebox.showinfo("Precision Adjusted", "FP8 Base requires bf16 for compatibility.\n\nMixed precision has been automatically set to 'bf16'.")
                    self.fp8_warning_shown = True

    def start_vram_monitor(self):
        if not PYNVML_AVAILABLE: self.vram_label_var.set("VRAM: pynvml not installed"); return
        try:
            pynvml.nvmlInit(); self.monitoring_active = True; self.peak_vram = 0
            self.vram_thread = threading.Thread(target=self.vram_monitor_loop, daemon=True); self.vram_thread.start()
        except pynvml.NVMLError: self.vram_label_var.set(f"VRAM: NVML Error")

    def stop_vram_monitor(self):
        self.monitoring_active = False
        if PYNVML_AVAILABLE:
            try: pynvml.nvmlShutdown()
            except pynvml.NVMLError: pass

    def vram_monitor_loop(self):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            while self.monitoring_active:
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_gb = info.used / (1024**3)
                if used_gb > self.peak_vram: self.peak_vram = used_gb
                self.root.after(0, self.update_vram_display, used_gb, self.peak_vram, info.total / (1024**3))
                time.sleep(1)
        except pynvml.NVMLError: self.root.after(0, lambda: self.vram_label_var.set("VRAM: Monitoring Error"))

    def update_vram_display(self, used, peak, total):
        self.vram_label_var.set(f"VRAM: {used:.2f} GB / {total:.2f} GB")
        self.peak_vram_label_var.set(f"Peak VRAM: {peak:.2f} GB")
        
    def update_loss_graph(self, step=None, loss_value=None):
        if not MATPLOTLIB_AVAILABLE: return
        if step is not None and loss_value is not None: self.loss_data.append((step, loss_value))
        self.ax.clear(); self.setup_graph_style()
        if self.loss_data:
            steps, losses = zip(*self.loss_data)
            self.ax.plot(steps, losses, color='#68bcece8')
        self.canvas.draw()
        
    def update_progress_bar(self, current, total):
        percentage = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(percentage)
        self.progress_label_var.set(f"Epoch {current} of {total}" if total > 0 else "Epochs complete")
            
    def run_process(self, command, on_complete=None, output_widget=None):
        if output_widget is None: output_widget = self.output_text
        self.start_btn.config(state="disabled"); self.stop_btn.config(state="normal")
        self.last_line_was_progress = False
        command_display = ' '.join(f'"{part}"' if ' ' in part else part for part in command)
        output_widget.insert(tk.END, f"\n--- Running command ---\n{command_display}\n\n")

        try:
            env = os.environ.copy(); env['PYTHONUNBUFFERED'] = '1'; env['PYTHONUTF8'] = '1'
            project_root = os.getcwd(); src_path = os.path.join(project_root, 'src')
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"

            self.current_process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=project_root,
                encoding='utf-8', errors='replace', bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0, env=env
            )
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not find '{e.filename}'. Is it in your system's PATH or venv?")
            self.stop_all_activity(); return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {e}")
            self.stop_all_activity(); return
        
        threading.Thread(target=self.read_output, args=(on_complete, output_widget), daemon=True).start()
    
    def stop_all_activity(self):
        self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        self.stop_vram_monitor(); self.current_process = None
        self.update_button_states()

    def process_console_output(self, line, output_widget):
        is_progress_line = line.endswith('\r')
        clean_line = line.strip()
        if is_progress_line:
            if self.last_line_was_progress: output_widget.delete("end-2l", "end-1l")
            output_widget.insert(tk.END, clean_line + '\n')
            self.last_line_was_progress = True
        else:
            output_widget.insert(tk.END, line)
            self.last_line_was_progress = False
        output_widget.see(tk.END)

    def read_output(self, on_complete, output_widget):
        if not self.current_process: 
            if on_complete: self.root.after(0, on_complete, -1); return
        try:
            buffer = ""
            while True:
                char = self.current_process.stdout.read(1)
                if not char and self.current_process.poll() is not None: break
                if not char: continue
                buffer += char
                if char in ('\n', '\r'):
                    self.root.after(0, self.process_console_output, buffer, output_widget)
                    if output_widget == self.output_text:
                        loss_match = re.search(r" loss=([\d\.]+)", buffer); step_match = re.search(r"(\d+)/\d+ \[", buffer)
                        epoch_match = re.search(r"epoch\s*=\s*(\d+)\s*/\s*(\d+)", buffer, re.IGNORECASE)
                        if loss_match and step_match: self.root.after(0, self.update_loss_graph, int(step_match.group(1)), float(loss_match.group(1)))
                        if epoch_match: self.root.after(0, self.update_progress_bar, int(epoch_match.group(1)), int(epoch_match.group(2)))
                    buffer = ""
            if buffer: self.root.after(0, self.process_console_output, buffer, output_widget)
        except Exception as e:
            self.root.after(0, output_widget.insert, tk.END, f"\n[Read error] {e}\n")
        finally:
            return_code = self.current_process.wait() if self.current_process else -1
            self.current_process = None
            if on_complete: self.root.after(0, on_complete, return_code)

    def _run_next_command_in_sequence(self, return_code):
        if return_code != 0:
            self.output_text.insert(tk.END, f"\n--- Previous step failed with code {return_code}. Halting sequence. ---\n")
            self.stop_all_activity(); return
        if self.command_sequence:
            next_command = self.command_sequence.pop(0)
            self.run_process(next_command, self._run_next_command_in_sequence, self.output_text)
        else:
            self.output_text.insert(tk.END, f"\n--- All steps completed successfully. ---\n")
            self.stop_all_activity()

    def _check_logging_dependencies(self, log_with):
        if log_with in ["wandb", "all"]:
            try: import wandb
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install wandb"); return False
        if log_with in ["tensorboard", "all"]:
            try: import tensorboard
            except Exception: messagebox.showerror("Missing Dependency", "Please run: pip install tensorboard"); return False
        return True

    def start_training(self):
        self.update_button_states(); settings = self.get_settings()
        if not self._check_logging_dependencies(settings.get("log_with")): return
        if self.start_btn['state'] == 'disabled':
            messagebox.showerror("Validation Error", "Please fill all required fields and select at least one DiT model to train."); return
        self.loss_data.clear(); self.update_loss_graph(); self.start_vram_monitor()
        self.progress_var.set(0); self.progress_label_var.set("Starting sequence...")
        self.output_text.delete("1.0", tk.END); self.command_sequence = []
        python_executable = sys.executable or "python"
        
        if settings.get("recache_latents"):
            latents_cmd = [python_executable, "src/musubi_tuner/wan_cache_latents.py", "--dataset_config", settings["dataset_config"], "--vae", settings["vae_model"]]
            self.command_sequence.append(latents_cmd)
        if settings.get("recache_text"):
            text_cmd = [python_executable, "src/musubi_tuner/wan_cache_text_encoder_outputs.py", "--dataset_config", settings["dataset_config"], "--t5", settings["t5_model"]]
            self.command_sequence.append(text_cmd)
        
        training_commands = self.build_training_commands()
        if training_commands: self.command_sequence.extend(training_commands)
        if self.command_sequence: self._run_next_command_in_sequence(0)
        else: messagebox.showwarning("Warning", "No training or caching steps were selected."); self.stop_all_activity()

    def stop_training(self):
        if self.current_process:
            self.output_text.insert(tk.END, "\n⚠️ Terminating process and sequence...\n")
            self.command_sequence = [];
            try: self.current_process.terminate()
            except Exception: pass
            self.stop_all_activity()

    def build_training_commands(self):
        settings = self.get_settings(); commands = []
        train_low = settings.get("train_low_noise"); train_high = settings.get("train_high_noise")
        
        def build_single_command(is_high_noise_run, is_combined_run):
            def normalize_path(p): return p.replace(os.sep, '/') if isinstance(p, str) and p else p
            def add_arg(cmd_list, key, value, is_path=False):
                clean_value = str(value).strip()
                if clean_value not in (None, "", "False"):
                    cmd_list.extend([key, str(normalize_path(clean_value) if is_path else clean_value)] if clean_value not in (True, "True") else [key])

            command = ["accelerate", "launch", "--num_cpu_threads_per_process", "1", "src/musubi_tuner/wan_train_network.py"]
            add_arg(command, "--task", "t2v-A14B")
            if settings.get("is_i2v"): add_arg(command, "--i2v", True)
            add_arg(command, "--mixed_precision", settings.get("mixed_precision"))

            add_arg(command, "--vae", settings.get("vae_model"), is_path=True); add_arg(command, "--t5", settings.get("t5_model"), is_path=True)
            add_arg(command, "--clip", settings.get("clip_model"), is_path=True); add_arg(command, "--dataset_config", settings.get("dataset_config"), is_path=True)
            
            if is_combined_run:
                add_arg(command, "--dit", settings.get("dit_low_noise"), is_path=True)
                add_arg(command, "--dit_high_noise", settings.get("dit_high_noise"), is_path=True)
                add_arg(command, "--timestep_boundary", settings.get("timestep_boundary"))
            else:
                add_arg(command, "--dit", settings.get("dit_high_noise") if is_high_noise_run else settings.get("dit_low_noise"), is_path=True)
                add_arg(command, "--min_timestep", settings.get("min_timestep_high") if is_high_noise_run else settings.get("min_timestep_low"))
                add_arg(command, "--max_timestep", settings.get("max_timestep_high") if is_high_noise_run else settings.get("max_timestep_low"))

            dim = settings.get("network_dim_high") if is_high_noise_run and settings.get("network_dim_high") else settings.get("network_dim_low")
            alpha = settings.get("network_alpha_high") if is_high_noise_run and settings.get("network_alpha_high") else settings.get("network_alpha_low")
            add_arg(command, "--network_module", "networks.lora_wan"); add_arg(command, "--network_dim", dim); add_arg(command, "--network_alpha", alpha)
            
            attention = settings.get("attention_mechanism");
            if attention and attention != "none": command.append(f"--{attention}")
            add_arg(command, "--fp8_base", settings.get("fp8_base")); add_arg(command, "--fp8_scaled", settings.get("fp8_scaled")); add_arg(command, "--fp8_t5", settings.get("fp8_t5"))
            add_arg(command, "--optimizer_type", settings.get("optimizer_type")); add_arg(command, "--learning_rate", settings.get("learning_rate"))
            add_arg(command, "--gradient_checkpointing", settings.get("gradient_checkpointing")); add_arg(command, "--gradient_accumulation_steps", settings.get("gradient_accumulation_steps"))
            add_arg(command, "--max_data_loader_n_workers", settings.get("max_data_loader_n_workers")); add_arg(command, "--persistent_data_loader_workers", settings.get("persistent_data_loader_workers"))

            if is_combined_run and settings.get("offload_inactive_dit"): add_arg(command, "--offload_inactive_dit", True)
            else: add_arg(command, "--blocks_to_swap", settings.get("blocks_to_swap"))
            
            add_arg(command, "--timestep_sampling", settings.get("timestep_sampling")); add_arg(command, "--num_timestep_buckets", settings.get("num_timestep_buckets"))
            add_arg(command, "--discrete_flow_shift", settings.get("discrete_flow_shift")); add_arg(command, "--preserve_distribution_shape", settings.get("preserve_distribution_shape"))
            add_arg(command, "--optimizer_args", settings.get("optimizer_args"))
            
            lr_scheduler = settings.get("lr_scheduler")
            if lr_scheduler and lr_scheduler != "constant":
                add_arg(command, "--lr_scheduler", lr_scheduler)
                if lr_scheduler == "constant_with_warmup": add_arg(command, "--lr_warmup_steps", settings.get("lr_warmup_steps"))
                if lr_scheduler == "cosine_with_restarts": add_arg(command, "--lr_scheduler_num_cycles", settings.get("lr_scheduler_num_cycles"))
                add_arg(command, "--lr_scheduler_power", settings.get("lr_scheduler_power")); add_arg(command, "--lr_scheduler_min_lr_ratio", settings.get("lr_scheduler_min_lr_ratio"))

            add_arg(command, "--max_train_epochs", settings.get("max_train_epochs")); add_arg(command, "--save_every_n_epochs", settings.get("save_every_n_epochs"))
            add_arg(command, "--save_every_n_steps", settings.get("save_every_n_steps")); add_arg(command, "--seed", settings.get("seed"))
            
            suffix = ""
            if train_low and train_high and not is_combined_run: suffix = "_HighNoise" if is_high_noise_run else "_LowNoise"
            elif train_high: suffix = "_HighNoise"
            elif train_low: suffix = "_LowNoise"
            
            output_dir = Path(settings.get("output_dir")) / (settings.get("output_name") + suffix)
            output_name = settings.get("output_name") + suffix
            os.makedirs(output_dir, exist_ok=True)
            add_arg(command, "--output_dir", str(output_dir), is_path=True)
            add_arg(command, "--output_name", output_name)
            
            add_arg(command, "--save_state", settings.get("save_state")); add_arg(command, "--resume", settings.get("resume_path"), is_path=True)
            add_arg(command, "--network_weights", settings.get("network_weights"), is_path=True)
            
            log_with = settings.get("log_with")
            if log_with and log_with != "none":
                add_arg(command, "--log_with", log_with); add_arg(command, "--logging_dir", settings.get("logging_dir"), is_path=True); add_arg(command, "--log_prefix", settings.get("log_prefix"))
            return command
        
        is_separate_run = train_low and train_high and (settings.get("network_dim_high") or settings.get("network_alpha_high"))
        is_combined_run = train_low and train_high and not is_separate_run

        if is_separate_run:
            commands.append(build_single_command(is_high_noise_run=False, is_combined_run=False))
            commands.append(build_single_command(is_high_noise_run=True, is_combined_run=False))
        elif is_combined_run:
             commands.append(build_single_command(is_high_noise_run=True, is_combined_run=True))
        elif train_low:
             commands.append(build_single_command(is_high_noise_run=False, is_combined_run=False))
        elif train_high:
             commands.append(build_single_command(is_high_noise_run=True, is_combined_run=False))
        return commands

    def show_command(self):
        commands = self.build_training_commands()
        if commands:
            full_command_str = ""
            for i, command in enumerate(commands):
                command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
                full_command_str += f"--- Command {i+1} ---\n{command_str}\n\n"
            dialog = tk.Toplevel(self.root); dialog.title("Generated Command(s)"); dialog.geometry("800x400")
            text = tk.Text(dialog, wrap="word", font=("Consolas", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
            text.insert("1.0", full_command_str); text.config(state="disabled")
            try: self.root.clipboard_clear(); self.root.clipboard_append(full_command_str)
            except Exception: pass

    def start_conversion(self):
        lora_path = self.entries["convert_lora_path"].get()
        output_path = self.entries["convert_output_path"].get()
        precision = self.entries["convert_precision"].get()

        if not (lora_path and os.path.exists(lora_path) and output_path and precision):
            messagebox.showerror("Validation Error", "Please fill all fields for conversion correctly."); return
        
        self.convert_output_text.delete("1.0", tk.END)
        python_executable = sys.executable or "python"
        command = [python_executable, "src/musubi_tuner/convert_lora.py",
                   "--lora_path", lora_path, "--output_path", output_path, "--precision", precision]
        
        self.run_process(command, on_complete=self.on_conversion_complete, output_widget=self.convert_output_text)

    def on_conversion_complete(self, return_code):
        if return_code == 0:
            self.convert_output_text.insert(tk.END, "\n--- Conversion completed successfully. ---")
        else:
            self.convert_output_text.insert(tk.END, f"\n--- Conversion failed with code {return_code}. ---")
        self.stop_all_activity()

    def run_accelerate_config(self):
        # --- ADDED --- Logic to run accelerate config in a new terminal
        try:
            python_executable = Path(sys.executable)
            accelerate_path = python_executable.parent / "accelerate"
            if sys.platform == "win32":
                accelerate_path = accelerate_path.with_suffix(".exe")

            if not accelerate_path.exists():
                accelerate_path = "accelerate" 

            command = f'"{accelerate_path}" config'

            if sys.platform == "win32":
                subprocess.Popen(f'start cmd /k {command}', shell=True)
            elif sys.platform == "darwin":
                script = f'tell application "Terminal" to do script "{command}"'
                subprocess.Popen(['osascript', '-e', script])
            else: 
                try:
                    subprocess.Popen(['x-terminal-emulator', '-e', command])
                except FileNotFoundError:
                    messagebox.showerror("Error", "Could not find a default terminal. Please run 'accelerate config' manually in your terminal.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch 'accelerate config': {e}\nPlease run it manually in your activated virtual environment.")

    def on_closing(self):
        self._save_settings_to_file("last_settings.json")
        if self.current_process and messagebox.askokcancel("Quit", "A process is running. Stop it and quit?"):
            self.stop_training()
        self.stop_vram_monitor(); self.root.destroy()
        
if __name__ == "__main__":
    if not PYNVML_AVAILABLE: print("WARNING: pynvml not found. VRAM monitoring disabled. Run 'pip install pynvml'.")
    if not MATPLOTLIB_AVAILABLE: print("WARNING: matplotlib not found. Live graph disabled. Run 'pip install matplotlib'.")
    root = tk.Tk()
    app = MusubiTunerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()