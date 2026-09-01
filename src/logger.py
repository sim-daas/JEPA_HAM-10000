import os
import json
import time
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class RunLogger:
    def __init__(self, paradigm: str):
        """
        Initializes the logger and TensorBoard writer for a given paradigm
        (e.g., 'lora', 'full_finetune', 'frozen_probe').
        """
        self.paradigm = paradigm
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join("logs", self.paradigm, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.metrics = {
            "hparams": {},
            "epochs": [],
            "folds": {},
            "peak_gpu_memory_MB": 0.0,
            "total_time_seconds": 0.0
        }
        self.run_start_time = time.time()
        self.epoch_start_time = None
        
        print(f"[{self.paradigm.upper()}] Logging initialized at {self.log_dir}")
        
    def log_hparams(self, hparams: dict):
        self.metrics["hparams"].update(hparams)
        with open(os.path.join(self.log_dir, "hparams.json"), "w") as f:
            json.dump(self.metrics["hparams"], f, indent=4)
            
    def log_step(self, split: str, metrics_dict: dict, global_step: int):
        """Logs metrics like loss and learning rate at each step."""
        for k, v in metrics_dict.items():
            self.writer.add_scalar(f"{split}/{k}", v, global_step)
            
    def epoch_start(self):
        self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def log_epoch(self, fold: int, epoch: int, metrics_dict: dict):
        """Logs aggregated metrics at the end of an epoch."""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        
        # Track peak GPU memory
        peak_mem = 0.0
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if peak_mem > self.metrics["peak_gpu_memory_MB"]:
                self.metrics["peak_gpu_memory_MB"] = peak_mem
                
        # Write to TensorBoard
        for k, v in metrics_dict.items():
            self.writer.add_scalar(f"fold_{fold}/{k}", v, epoch)
            
        # Store in dict for JSON
        epoch_data = {
            "fold": fold,
            "epoch": epoch,
            "time_seconds": epoch_time,
            "peak_mem_MB": peak_mem,
            **metrics_dict
        }
        self.metrics["epochs"].append(epoch_data)
        
    def log_fold_result(self, fold: int, test_f1: float):
        self.metrics["folds"][f"fold_{fold}"] = test_f1
        
    def finish(self, final_metrics: dict = None):
        """Called at the end of the full run."""
        self.metrics["total_time_seconds"] = time.time() - self.run_start_time
        if final_metrics:
            self.metrics.update(final_metrics)
            
        # Save JSON
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
            
        self.writer.close()
        print(f"[{self.paradigm.upper()}] Run completed. Peak GPU Mem: {self.metrics['peak_gpu_memory_MB']:.2f} MB")
        print(f"[{self.paradigm.upper()}] Logs saved to {self.log_dir}")
