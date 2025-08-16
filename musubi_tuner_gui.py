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
                         font=("Calibri", "10", "normal"), wraplength=300)
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
        try:
            style.theme_use('clam')
        except Exception:
            pass
        
        style.configure('.', background=BG_COLOR, foreground=TEXT_COLOR, font=('Calibri', 9))
        style.configure('TLabel', font=('Calibri', 10))
        style.configure('TFrame', background=BG_COLOR)
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
        style.configure('TCheckbutton', font=('Calibri', 10))
        style.configure('Title.TLabel', font=('Calibri', 16, 'bold'))
        style.configure('Status.TLabel', font=('Calibri', 11, 'bold'))
        
        style.configure('TProgressbar', thickness=20, background=SELECT_BG_COLOR, troughcolor=FIELD_BG_COLOR)
        
        style.configure('Invalid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=ERROR_BORDER, foreground=TEXT_COLOR, relief='solid', borderwidth=1)
        style.configure('Valid.TEntry', fieldbackground=FIELD_BG_COLOR, bordercolor=BORDER_COLOR, foreground=TEXT_COLOR, relief='solid', borderwidth=1)

    def create_interface(self):
        canvas = tk.Canvas(self.root, bg='#2B2B2B', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True); scrollbar.pack(side="right", fill="y")
        self.root.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        main_frame = ttk.Frame(scrollable_frame); main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Label(main_frame, text="Musubi Tuner - WAN 2.2 LoRA Training", style='Title.TLabel').pack(pady=(0, 20), anchor='w')
        self.create_settings_buttons(main_frame)
        
        self.notebook = ttk.Notebook(main_frame); self.notebook.pack(fill="both", expand=True, pady=(10, 0))
        
        self.create_model_paths_tab()
        self.create_training_params_tab()
        self.create_advanced_tab()
        self.create_run_monitor_tab()
    
    def create_settings_buttons(self, parent):
        button_frame = ttk.Frame(parent); button_frame.pack(fill="x", pady=(0, 10), anchor='w')
        ttk.Button(button_frame, text="Load Settings", command=self.load_settings).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Save Settings", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.load_default_settings).pack(side="left", padx=5)

    def _add_widget(self, parent, key, label, tooltip, kind='entry', options=None, is_required=False, validate_num=False, is_path=False, is_dir=False, default_val=False, command=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=(5, 8))
        if kind != 'checkbox':
            ttk.Label(frame, text=label).pack(anchor="w")
        
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
            widget.pack(fill="x", pady=(2, 0))
            widget.bind("<MouseWheel>", lambda e: "break")
        elif kind == 'checkbox':
            var = tk.BooleanVar(value=default_val)
            def chained_command():
                if command and callable(command):
                    command()
                self.update_button_states()
            widget = ttk.Checkbutton(frame, text=label, variable=var, command=chained_command)
            widget.var = var
            widget.pack(anchor="w", padx=5, pady=2)
        else:
            vcmd = (self.root.register(self.validate_number), '%P') if validate_num else None
            widget = ttk.Entry(frame, validate="key", validatecommand=vcmd)
            widget.pack(fill="x", pady=(2, 0))

        if tooltip: ToolTip(widget, tooltip)
        self.entries[key] = widget
        widget.is_required = is_required
        widget.is_path = is_path
        
        if isinstance(widget, ttk.Entry):
            widget.bind("<FocusOut>", self.update_button_states)
            widget.bind("<KeyRelease>", self.update_button_states)
        
        return widget
    
    def create_model_paths_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Model Paths & Dataset")
        
        dataset_frame = ttk.LabelFrame(frame, text="Dataset Configuration"); dataset_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(dataset_frame, "dataset_config", "Dataset Config (TOML):", "Path to .toml dataset configuration file.", kind='path_entry', options=[("TOML files", "*.toml")], is_required=True, is_path=True)
        
        dit_frame = ttk.LabelFrame(frame, text="DiT Model Selection"); dit_frame.pack(fill="x", padx=10, pady=10)
        
        high_noise_frame = ttk.LabelFrame(dit_frame, text="High Noise Model (Timesteps: 875-1000)"); high_noise_frame.pack(fill="x", padx=5, pady=5)
        self._add_widget(high_noise_frame, "train_high_noise", "Train High Noise Model", "Enable to train the high noise model.", kind='checkbox')
        self._add_widget(high_noise_frame, "dit_high_noise", "DiT High Noise Model Path:", "Path to the high noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)

        low_noise_frame = ttk.LabelFrame(dit_frame, text="Low Noise Model (Timesteps: 0-875)"); low_noise_frame.pack(fill="x", padx=5, pady=(5,10))
        self._add_widget(low_noise_frame, "train_low_noise", "Train Low Noise Model", "Enable to train the low noise model. If both are selected, this becomes the base --dit model.", kind='checkbox')
        self._add_widget(low_noise_frame, "dit_low_noise", "DiT Low Noise Model Path:", "Path to the low noise DiT model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)

        models_frame = ttk.LabelFrame(frame, text="Other Model Paths"); models_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(models_frame, "vae_model", "VAE Model:", "Path to VAE model (.safetensors or .pt).", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        self._add_widget(models_frame, "clip_model", "CLIP Model (Optional):", "Path to optional CLIP model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_path=True)
        self._add_widget(models_frame, "t5_model", "T5 Text Encoder:", "Path to T5 text encoder model.", kind='path_entry', options=[("Model files", "*.safetensors *.pt")], is_required=True, is_path=True)
        
        output_frame = ttk.LabelFrame(frame, text="Output Configuration"); output_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(output_frame, "output_dir", "Output Directory:", "Directory to save trained LoRAs.", kind='path_entry', is_dir=True, is_required=True, is_path=True)
        self._add_widget(output_frame, "output_name", "Output Name:", "Filename for output LoRA (no extension).", is_required=True)

    def create_training_params_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Training Parameters")
        basic_frame = ttk.LabelFrame(frame, text="Basic Training Parameters"); basic_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(basic_frame, "learning_rate", "Learning Rate:", "Initial learning rate (e.g., 1e-4).", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "max_train_epochs", "Max Train Epochs:", "Total training epochs.", is_required=True, validate_num=True)
        self._add_widget(basic_frame, "save_every_n_epochs", "Save Every N Epochs:", "Frequency of saving checkpoints.", validate_num=True)
        self._add_widget(basic_frame, "seed", "Seed:", "Number for reproducible training.", validate_num=True)
        
        network_frame = ttk.LabelFrame(frame, text="LoRA Network Parameters"); network_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(network_frame, "network_dim", "Network Dimension (Rank):", "Intrinsic rank of LoRA (e.g., 32, 64).", is_required=True, validate_num=True)
        self._add_widget(network_frame, "network_alpha", "Network Alpha:", "LoRA learning rate scaler.", is_required=True, validate_num=True)
        
        optimizer_frame = ttk.LabelFrame(frame, text="Optimizer Settings"); optimizer_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(optimizer_frame, "optimizer_type", "Optimizer Type:", "'adamw8bit' is a good default.", kind='combobox', options=["adamw", "adamw8bit", "adafactor", "lion", "prodigy"])
        self._add_widget(optimizer_frame, "optimizer_args", "Optimizer Args:", "Additional args (e.g., weight_decay=0.01).")
        
        lr_frame = ttk.LabelFrame(frame, text="Learning Rate Scheduler"); lr_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(lr_frame, "lr_scheduler", "LR Scheduler:", "Algorithm to adjust learning rate.", kind='combobox', options=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"])
        self._add_widget(lr_frame, "lr_scheduler_power", "LR Scheduler Power:", "Exponent for polynomial scheduler.", validate_num=True)
        self._add_widget(lr_frame, "lr_scheduler_min_lr_ratio", "Min LR Ratio:", "Minimum learning rate ratio.", validate_num=True)

    def create_advanced_tab(self):
        frame = ttk.Frame(self.notebook); self.notebook.add(frame, text="Advanced Settings")
        memory_frame = ttk.LabelFrame(frame, text="Memory & Performance"); memory_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(memory_frame, "mixed_precision", "Mixed Precision:", "Use 'fp16' or 'bf16' to save VRAM.", kind='combobox', options=["no", "fp16", "bf16"])
        self._add_widget(memory_frame, "gradient_checkpointing", "Gradient Checkpointing", "Drastically reduces VRAM. Highly recommended.", kind='checkbox', default_val=True)
        self._add_widget(memory_frame, "persistent_data_loader_workers", "Persistent Data Loader Workers", "Keeps data loader workers alive between epochs.", kind='checkbox')
        self._add_widget(memory_frame, "gradient_accumulation_steps", "Gradient Accumulation Steps:", "Simulate a larger batch size", validate_num=True)
        self._add_widget(memory_frame, "max_data_loader_n_workers", "Max Data Loader Workers:", "CPU threads to load data. '2' is safe.", validate_num=True)
        self._add_widget(memory_frame, "offload_inactive_dit", "Offload Inactive DiT Model", "When training Wan2.2 high and low models, you can use --offload_inactive_dit to offload the inactive DiT model to the CPU, which can save VRAM (AUTOMATICALLY DISABLE --blocks_to_swap).", kind='checkbox')
        self._add_widget(memory_frame, "blocks_to_swap", "Blocks to Swap:", "DiT blocks to offload to CPU. Disabled if Offload Inactive DiT is used.", validate_num=True)
        
        flow_frame = ttk.LabelFrame(frame, text="Flow Matching Parameters"); flow_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(flow_frame, "timestep_sampling", "Timestep Sampling:", "Method for selecting timesteps.", kind='combobox', options=["uniform", "shift"])
        self._add_widget(flow_frame, "discrete_flow_shift", "Discrete Flow Shift:", "Shift value for 'shift' sampling.", validate_num=True)
        self._add_widget(flow_frame, "preserve_distribution_shape", "Preserve Distribution Shape", "Helps prevent mode collapse during single-model training.", kind='checkbox')

        attention_frame = ttk.LabelFrame(frame, text="Attention Mechanism"); attention_frame.pack(fill="x", padx=10, pady=10)
        self.entries['attention_mechanism'] = tk.StringVar(value="xformers")
        attention_options = [("None", "none"), ("xFormers", "xformers"), ("Flash Attention", "flash_attn"), ("SDPA", "sdpa")]
        for text, value in attention_options:
            rb = ttk.Radiobutton(attention_frame, text=text, variable=self.entries['attention_mechanism'], value=value)
            rb.pack(anchor="w", padx=5, pady=2); ToolTip(rb, f"Optimized attention to save VRAM and speed up.")

        logging_frame = ttk.LabelFrame(frame, text="Logging (TensorBoard / W&B)"); logging_frame.pack(fill="x", padx=10, pady=10)
        log_with_widget = self._add_widget(logging_frame, "log_with", "Log With:", "Enable logging with TensorBoard or Weights & Biases.", kind='combobox', options=["none", "tensorboard", "wandb", "all"])
        log_with_widget.bind('<<ComboboxSelected>>', self.update_button_states)
        self._add_widget(logging_frame, "logging_dir", "Logging Directory:", "Directory to save logs. Required if 'Log With' is not 'none'.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(logging_frame, "log_prefix", "Log Prefix:", "Optional prefix for log filenames.")

        other_frame = ttk.LabelFrame(frame, text="Other Options"); other_frame.pack(fill="x", padx=10, pady=10)
        fp8_frame = ttk.Frame(other_frame); fp8_frame.pack(fill='x')
        self._add_widget(fp8_frame, "fp8_base", "FP8 Base", "Use FP8 precision. Automatically sets mixed precision to 'bf16'.", kind='checkbox', command=self._handle_fp8_precision_conflict)
        self._add_widget(fp8_frame, "fp8_scaled", "FP8 Scaled", "Use scaled FP8 training.", kind='checkbox')
        self._add_widget(other_frame, "save_state", "Save State", "Save training state to allow resuming later.", kind='checkbox', default_val=True)
        
        resume_frame = ttk.LabelFrame(frame, text="Resume Training"); resume_frame.pack(fill="x", padx=10, pady=10)
        self._add_widget(resume_frame, "resume_path", "Resume from State:", "Path to a saved state folder.", kind='path_entry', is_dir=True, is_path=True)
        self._add_widget(resume_frame, "network_weights", "Network Weights:", "Load pre-trained LoRA weights.", kind='path_entry', options=[("Weight files", "*.safetensors")], is_path=True)

    def create_run_monitor_tab(self):
        tab_frame = ttk.Frame(self.notebook); self.notebook.add(tab_frame, text="Run & Monitor")
        top_pane = ttk.Frame(tab_frame); top_pane.pack(fill='x', padx=10, pady=10)
        controls_frame = ttk.LabelFrame(top_pane, text="Controls & Caching"); controls_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # --- LED Status Indicator ---
        self.run_status_var = tk.StringVar(value="⚪ New Training RUN")
        self.run_status_label = ttk.Label(controls_frame, textvariable=self.run_status_var, style='Status.TLabel')
        self.run_status_label.pack(pady=5, padx=10)

        cache_opts_frame = ttk.Frame(controls_frame)
        cache_opts_frame.pack(pady=5, padx=10, fill='x')
        self._add_widget(cache_opts_frame, "recache_latents", "Re-cache Latents Before Training", "If checked, forces regeneration of latent cache.", kind='checkbox')
        self._add_widget(cache_opts_frame, "recache_text", "Re-cache Text Encoders Before Training", "If checked, forces regeneration of text encoder cache.", kind='checkbox')
        
        train_button_frame = ttk.Frame(controls_frame); train_button_frame.pack(pady=10, padx=10, fill='x')
        self.start_btn = ttk.Button(train_button_frame, text="Start Training", command=self.start_training); self.start_btn.pack(side="left", padx=(0, 5), expand=True, fill='x')
        self.stop_btn = ttk.Button(train_button_frame, text="Stop Training", command=self.stop_training, state="disabled"); self.stop_btn.pack(side="left", padx=5, expand=True, fill='x')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(controls_frame, variable=self.progress_var, style='TProgressbar')
        self.progress_bar.pack(pady=(5, 5), padx=10, fill='x')
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
    
    def setup_graph_style(self):
        self.fig.patch.set_facecolor('#2B2B2B')
        self.ax.set_facecolor('#3C3F41')
        self.ax.tick_params(axis='x', colors='white'); self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white'); self.ax.spines['top'].set_color('white') 
        self.ax.spines['right'].set_color('white'); self.ax.spines['left'].set_color('white')
        self.ax.yaxis.label.set_color('white'); self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_xlabel("Steps"); self.ax.set_ylabel("Loss")
        self.canvas.draw()
    
    def validate_number(self, value):
        if value in ("", "-"): return True
        try: float(value); return True
        except ValueError: return False

    def update_button_states(self, event=None):
        try:
            # Update resume status indicator
            if self.entries["resume_path"].get():
                self.run_status_var.set("🟢 Resuming Training RUN")
            else:
                self.run_status_var.set("⚪ New Training RUN")

            train_both = self.entries["train_high_noise"].var.get() and self.entries["train_low_noise"].var.get()
            offload_widget = self.entries["offload_inactive_dit"]
            blocks_to_swap_widget = self.entries["blocks_to_swap"]

            offload_widget.config(state="normal" if train_both else "disabled")
            if not train_both:
                offload_widget.var.set(False) 

            is_offloading = offload_widget.var.get() and train_both
            if blocks_to_swap_widget.cget('state') != ('disabled' if is_offloading else 'normal'):
                blocks_to_swap_widget.config(state="disabled" if is_offloading else "normal")
                if is_offloading:
                    blocks_to_swap_widget.delete(0, tk.END)
        except (KeyError, AttributeError):
            pass

        all_valid = True
        train_high = self.entries["train_high_noise"].var.get()
        train_low = self.entries["train_low_noise"].var.get()
        self.entries["dit_high_noise"].is_required = train_high
        self.entries["dit_low_noise"].is_required = train_low

        log_with = self.entries["log_with"].get()
        is_logging_enabled = log_with != "none"
        self.entries["logging_dir"].is_required = is_logging_enabled

        for widget in self.entries.values():
            if not isinstance(widget, ttk.Entry): continue
            
            is_valid = True
            if widget.cget('state') == 'disabled':
                widget.config(style="Valid.TEntry")
                continue

            if getattr(widget, 'is_required', False):
                value = widget.get()
                if not value: is_valid = False
                elif getattr(widget, 'is_path', False) and not os.path.exists(value): is_valid = False
            
            style = "Valid.TEntry" if is_valid else "Invalid.TEntry"
            widget.config(style=style)
            if not is_valid: all_valid = False
        
        if not (train_high or train_low):
            all_valid = False

        self.start_btn.config(state="normal" if all_valid else "disabled")
        try:
            self.entries["recache_latents"].config(state="normal" if all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "vae_model"]) else "disabled")
            self.entries["recache_text"].config(state="normal" if all(self.entries[key].get() and os.path.exists(self.entries[key].get()) for key in ["dataset_config", "t5_model"]) else "disabled")
        except (AttributeError, KeyError):
             pass

    def get_settings(self):
        settings = {}
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
                else:
                    if hasattr(widget, 'set'): widget.set(value)
                    else: widget.delete(0, tk.END); widget.insert(0, str(value))
        self.update_button_states()

    def load_default_settings(self):
        defaults = {
            "dataset_config": "", "dit_high_noise": "", "dit_low_noise": "",
            "train_high_noise": True, "train_low_noise": False,
            "vae_model": "", "clip_model": "", "t5_model": "",
            "output_dir": "", "output_name": "my-lora",
            "learning_rate": "2e-4", "max_train_epochs": "16", "save_every_n_epochs": "1", "seed": "42",
            "network_dim": "32", "network_alpha": "16", "optimizer_type": "adamw8bit",
            "optimizer_args": "weight_decay=0.1", "lr_scheduler": "cosine",
            "mixed_precision": "fp16", "gradient_accumulation_steps": "1",
            "max_data_loader_n_workers": "2", "blocks_to_swap": "10", "timestep_sampling": "shift",
            "discrete_flow_shift": "3.0", "preserve_distribution_shape": True,
            "gradient_checkpointing": True, "persistent_data_loader_workers": True, "save_state": True, 
            "fp8_base": True, "fp8_scaled": True,
            "offload_inactive_dit": False,
            "attention_mechanism": "xformers", "resume_path": "", "network_weights": "",
            "log_with": "none", "logging_dir": "", "log_prefix": "",
            "recache_latents": False, "recache_text": False
        }
        self.set_values(defaults)
        
    def _save_settings_to_file(self, filepath):
        try:
            with open(filepath, "w") as f:
                json.dump(self.get_settings(), f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings to {filepath}: {e}")
            return False

    def save_settings(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            if self._save_settings_to_file(file_path):
                messagebox.showinfo("Success", "Settings saved successfully!")

    def load_settings(self, filepath=None):
        if filepath is None:
            filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    settings = json.load(f)
                self.set_values(settings)
                if not filepath.endswith("last_settings.json"):
                    messagebox.showinfo("Success", "Settings loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load settings: {e}")

    def _load_last_settings(self):
        self.load_settings(filepath="last_settings.json")
    
    def _handle_fp8_precision_conflict(self):
        if self.entries["fp8_base"].var.get():
            precision_widget = self.entries["mixed_precision"]
            if precision_widget.get() != 'bf16':
                precision_widget.set('bf16')
                if not self.fp8_warning_shown:
                    messagebox.showinfo("Precision Adjusted", "FP8 Base requires bf16 for compatibility with this model type.\n\nMixed precision has been automatically set to 'bf16'.")
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
        except pynvml.NVMLError:
            self.root.after(0, lambda: self.vram_label_var.set("VRAM: Monitoring Error"))

    def update_vram_display(self, used, peak, total):
        self.vram_label_var.set(f"VRAM: {used:.2f} GB / {total:.2f} GB")
        self.peak_vram_label_var.set(f"Peak VRAM: {peak:.2f} GB")
        
    def update_loss_graph(self, loss_value=None):
        if not MATPLOTLIB_AVAILABLE: return
        if loss_value is not None: self.loss_data.append(loss_value)
        self.ax.clear(); self.setup_graph_style()
        if self.loss_data: self.ax.plot(self.loss_data)
        self.canvas.draw()
        
    def update_progress_bar(self, current, total):
        percentage = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(percentage)
        self.progress_label_var.set(f"Epoch {current} of {total}" if total > 0 else "Epochs complete")
            
    def run_process(self, command, on_complete=None):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.last_line_was_progress = False
        command_display_list = [f'"{part}"' if ' ' in part else part for part in command]
        self.output_text.insert(tk.END, f"\n--- Running command ---\n{' '.join(command_display_list)}\n\n")

        try:
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONUTF8'] = '1'
            project_root = os.getcwd()
            src_path = os.path.join(project_root, 'src')
            existing = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{existing}" if existing else src_path

            self.current_process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=project_root,
                encoding='utf-8',
                errors='replace',
                bufsize=1, # Line-buffered
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0,
                env=env
            )
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Could not find '{e.filename}'. Is it in your system's PATH or venv?")
            self.stop_all_activity()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start process: {e}")
            self.stop_all_activity()
            return
        
        threading.Thread(target=self.read_output, args=(on_complete,), daemon=True).start()
    
    def stop_all_activity(self):
        self.start_btn.config(state="normal"); self.stop_btn.config(state="disabled")
        self.stop_vram_monitor(); self.current_process = None
        self.update_button_states()

    def process_console_output(self, line):
        """Intelligently appends or replaces lines in the console widget."""
        # A line ending with \r is a progress bar update.
        is_progress_line = line.endswith('\r')
        clean_line = line.strip()

        if is_progress_line:
            if self.last_line_was_progress:
                # If the last line was also a progress bar, delete it before inserting the new one.
                self.output_text.delete("end-2l", "end-1l")
            self.output_text.insert(tk.END, clean_line + '\n')
            self.last_line_was_progress = True
        else:
            self.output_text.insert(tk.END, line)
            self.last_line_was_progress = False
        
        self.output_text.see(tk.END)

    def read_output(self, on_complete):
        if not self.current_process: 
            if on_complete: self.root.after(0, on_complete, -1)
            return
        try:
            # Using read(1) to process character by character to correctly handle \r
            buffer = ""
            while True:
                char = self.current_process.stdout.read(1)
                if not char and self.current_process.poll() is not None:
                    break
                if not char:
                    continue
                
                buffer += char
                if char in ('\n', '\r'):
                    # Send the complete line (or progress update) for processing
                    self.root.after(0, self.process_console_output, buffer)
                    
                    # Parse for metrics regardless of line ending
                    loss_match = re.search(r"loss=([\d.]+)", buffer)
                    if loss_match: self.root.after(0, self.update_loss_graph, float(loss_match.group(1)))
                    epoch_match = re.search(r"epoch\s*=\s*(\d+)\s*/\s*(\d+)", buffer, re.IGNORECASE)
                    if epoch_match: self.root.after(0, self.update_progress_bar, int(epoch_match.group(1)), int(epoch_match.group(2)))
                    
                    buffer = "" # Reset buffer
            
            # Process any remaining data in the buffer
            if buffer:
                self.root.after(0, self.process_console_output, buffer)

        except Exception as e:
            self.root.after(0, self.output_text.insert, tk.END, f"\n[Read error] {e}\n")
        finally:
            try:
                return_code = self.current_process.wait()
            except Exception:
                return_code = -1
            self.current_process = None
            if on_complete: self.root.after(0, on_complete, return_code)


    def _run_next_command_in_sequence(self, return_code):
        if return_code != 0:
            self.output_text.insert(tk.END, f"\n--- Previous step failed with code {return_code}. Halting sequence. ---\n")
            self.stop_all_activity()
            return
        
        if self.command_sequence:
            next_command = self.command_sequence.pop(0)
            self.run_process(next_command, self._run_next_command_in_sequence)
        else:
            self.output_text.insert(tk.END, f"\n--- All steps completed successfully. ---\n")
            self.stop_all_activity()

    def _check_logging_dependencies(self, log_with):
        if log_with in ["wandb", "all"]:
            try: import wandb
            except Exception:
                messagebox.showerror("Missing Dependency", "Please run: pip install wandb"); return False
        if log_with in ["tensorboard", "all"]:
            try: import tensorboard
            except Exception:
                 messagebox.showerror("Missing Dependency", "Please run: pip install tensorboard"); return False
        return True

    def start_training(self):
        self.update_button_states()
        settings = self.get_settings()
        if not self._check_logging_dependencies(settings.get("log_with")): return
        
        if self.start_btn['state'] == 'disabled':
            messagebox.showerror("Validation Error", "Please fill all required fields and select a DiT model."); return
        
        self.loss_data.clear(); self.update_loss_graph(); self.start_vram_monitor()
        self.progress_var.set(0); self.progress_label_var.set("Starting sequence...")
        self.output_text.delete("1.0", tk.END)
        self.command_sequence = []

        python_executable = sys.executable or "python"
        
        if settings.get("recache_latents"):
            cache_latents_script = "src/musubi_tuner/wan_cache_latents.py".replace(os.sep, '/')
            latents_cmd = [python_executable, cache_latents_script]
            latents_cmd.extend(["--dataset_config", settings["dataset_config"].replace(os.sep, '/')])
            latents_cmd.extend(["--vae", settings["vae_model"].replace(os.sep, '/')])
            self.command_sequence.append(latents_cmd)
            
        if settings.get("recache_text"):
            cache_text_script = "src/musubi_tuner/wan_cache_text_encoder_outputs.py".replace(os.sep, '/')
            text_cmd = [python_executable, cache_text_script]
            text_cmd.extend(["--dataset_config", settings["dataset_config"].replace(os.sep, '/')])
            text_cmd.extend(["--t5", settings["t5_model"].replace(os.sep, '/')])
            self.command_sequence.append(text_cmd)
        
        training_command = self.build_training_command()
        if training_command:
            self.command_sequence.append(training_command)

        if self.command_sequence:
            first_command = self.command_sequence.pop(0)
            self.run_process(first_command, self._run_next_command_in_sequence)
        else:
            messagebox.showwarning("Warning", "No training or caching steps were selected.")
            self.stop_all_activity()

    def stop_training(self):
        if self.current_process:
            self.output_text.insert(tk.END, "\n⚠️ Terminating process and sequence...\n")
            self.command_sequence = []
            try:
                self.current_process.terminate()
            except Exception:
                pass
            self.stop_all_activity()

    def build_training_command(self):
        settings = self.get_settings()
        
        def normalize_path(p):
            if isinstance(p, str) and p:
                return p.replace(os.sep, '/')
            return p

        def add_arg(command_list, key, value, is_path=False):
            if value not in (None, "", False):
                current_value = normalize_path(value) if is_path else value
                command_list.append(key)
                if current_value is not True:
                    command_list.append(str(current_value))

        command = ["accelerate", "launch", "--num_cpu_threads_per_process", "1"]
        script_path = "src/musubi_tuner/wan_train_network.py".replace(os.sep, '/')
        command.append(script_path)
        
        train_high = settings.get("train_high_noise")
        train_low = settings.get("train_low_noise")
        mixed_precision = settings.get("mixed_precision")
        if mixed_precision and mixed_precision != "no":
            add_arg(command, "--mixed_precision", mixed_precision)
        
        add_arg(command, "--task", "t2v-A14B")
        dit_high_path = settings.get("dit_high_noise")
        dit_low_path = settings.get("dit_low_noise")
        if train_high and train_low:
            add_arg(command, "--dit", dit_low_path, is_path=True)
            add_arg(command, "--dit_high_noise", dit_high_path, is_path=True)
        elif train_high:
            add_arg(command, "--dit", dit_high_path, is_path=True)
        elif train_low:
            add_arg(command, "--dit", dit_low_path, is_path=True)
        else: return None
        add_arg(command, "--vae", settings.get("vae_model"), is_path=True)
        add_arg(command, "--t5", settings.get("t5_model"), is_path=True)
        add_arg(command, "--clip", settings.get("clip_model"), is_path=True)
        add_arg(command, "--dataset_config", settings.get("dataset_config"), is_path=True)
        attention = settings.get("attention_mechanism", "xformers")
        if attention and attention != "none": command.append(f"--{attention}")
        add_arg(command, "--fp8_base", settings.get("fp8_base"))
        add_arg(command, "--fp8_scaled", settings.get("fp8_scaled"))
        add_arg(command, "--optimizer_type", settings.get("optimizer_type"))
        add_arg(command, "--learning_rate", settings.get("learning_rate"))
        add_arg(command, "--gradient_checkpointing", settings.get("gradient_checkpointing"))
        add_arg(command, "--gradient_accumulation_steps", settings.get("gradient_accumulation_steps"))
        add_arg(command, "--max_data_loader_n_workers", settings.get("max_data_loader_n_workers"))
        add_arg(command, "--persistent_data_loader_workers", settings.get("persistent_data_loader_workers"))
        offload_inactive = settings.get("offload_inactive_dit") and train_high and train_low
        if offload_inactive: add_arg(command, "--offload_inactive_dit", True)
        else: add_arg(command, "--blocks_to_swap", settings.get("blocks_to_swap"))
        add_arg(command, "--network_module", "networks.lora_wan")
        add_arg(command, "--network_dim", settings.get("network_dim"))
        add_arg(command, "--network_alpha", settings.get("network_alpha"))
        add_arg(command, "--timestep_sampling", settings.get("timestep_sampling"))
        if train_high and not train_low:
            add_arg(command, "--min_timestep", "875"); add_arg(command, "--max_timestep", "1000")
        elif train_low and not train_high:
            add_arg(command, "--min_timestep", "0"); add_arg(command, "--max_timestep", "875")
        add_arg(command, "--discrete_flow_shift", settings.get("discrete_flow_shift"))
        add_arg(command, "--preserve_distribution_shape", settings.get("preserve_distribution_shape"))
        add_arg(command, "--optimizer_args", settings.get("optimizer_args"))
        lr_scheduler = settings.get("lr_scheduler")
        if lr_scheduler and lr_scheduler != "constant":
            add_arg(command, "--lr_scheduler", lr_scheduler)
            add_arg(command, "--lr_scheduler_power", settings.get("lr_scheduler_power"))
            add_arg(command, "--lr_scheduler_min_lr_ratio", settings.get("lr_scheduler_min_lr_ratio"))
        add_arg(command, "--max_train_epochs", settings.get("max_train_epochs"))
        add_arg(command, "--save_every_n_epochs", settings.get("save_every_n_epochs"))
        add_arg(command, "--seed", settings.get("seed"))
        add_arg(command, "--output_dir", settings.get("output_dir"), is_path=True)
        add_arg(command, "--output_name", settings.get("output_name"))
        add_arg(command, "--save_state", settings.get("save_state"))
        add_arg(command, "--resume", settings.get("resume_path"), is_path=True)
        add_arg(command, "--network_weights", settings.get("network_weights"), is_path=True)
        log_with = settings.get("log_with")
        if log_with and log_with != "none":
            add_arg(command, "--log_with", log_with)
            add_arg(command, "--logging_dir", settings.get("logging_dir"), is_path=True)
            add_arg(command, "--log_prefix", settings.get("log_prefix"))
        return command

    def show_command(self):
        command = self.build_training_command()
        if command:
            command_str = " ".join(f'"{arg}"' if " " in arg else arg for arg in command)
            dialog = tk.Toplevel(self.root); dialog.title("Generated Command"); dialog.geometry("800x300")
            text = tk.Text(dialog, wrap="word", font=("Consolas", 10)); text.pack(expand=True, fill="both", padx=10, pady=10)
            text.insert("1.0", command_str); text.config(state="disabled")
            try:
                self.root.clipboard_clear(); self.root.clipboard_append(command_str)
            except Exception:
                pass

    def on_closing(self):
        self._save_settings_to_file("last_settings.json")
        if self.current_process:
            if messagebox.askokcancel("Quit", "Training is in progress. Stop it and quit?"):
                self.stop_training()
        self.stop_vram_monitor()
        self.root.destroy()
        
if __name__ == "__main__":
    if not PYNVML_AVAILABLE: print("WARNING: pynvml not found. VRAM monitoring disabled. Run 'pip install pynvml'.")
    if not MATPLOTLIB_AVAILABLE: print("WARNING: matplotlib not found. Live graph disabled. Run 'pip install matplotlib'.")
    root = tk.Tk()
    app = MusubiTunerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()