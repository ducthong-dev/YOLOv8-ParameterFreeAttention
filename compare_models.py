"""
Comprehensive comparison and evaluation script for YOLOv8 variants:
Baseline, ECA, SimAM, and Hybrid models
"""

import torch
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from thop import profile, clever_format
import numpy as np

class ModelComparison:
    """Compare multiple YOLOv8 classification model variants"""
    
    def __init__(self, data_yaml, imgsz=224, device='0'):
        self.data_yaml = data_yaml
        self.imgsz = imgsz
        self.device = device
        self.results = []
        
    def evaluate_model(self, model_path, model_name, weights_path=None):
        """Evaluate a single model and collect metrics"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}\n")
        
        # Load model
        if weights_path:
            model = YOLO(weights_path)
        else:
            model = YOLO(model_path)
        
        # Get model info
        model_info = self._get_model_info(model)
        
        # Validation metrics
        val_metrics = model.val(data=self.data_yaml, imgsz=self.imgsz, device=self.device)
        
        # Inference speed test
        speed_metrics = self._benchmark_speed(model, runs=100)
        
        # Collect results
        result = {
            'Model': model_name,
            'Top1_Accuracy': float(val_metrics.top1),
            'Top5_Accuracy': float(val_metrics.top5),
            'Parameters': model_info['parameters'],
            'GFLOPs': model_info['gflops'],
            'Inference_Speed_ms': speed_metrics['mean_time'],
            'Inference_Speed_std': speed_metrics['std_time'],
            'FPS': speed_metrics['fps'],
            'Model_Size_MB': model_info['size_mb']
        }
        
        self.results.append(result)
        
        print(f"\nResults for {model_name}:")
        for key, value in result.items():
            if key != 'Model':
                print(f"  {key}: {value}")
        
        return result
    
    def _get_model_info(self, model):
        """Get model parameters and FLOPs"""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        dummy_input = torch.randn(1, 3, self.imgsz, self.imgsz).to(self.device)
        
        try:
            flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9
        except:
            gflops = 0.0
        
        # Model size
        param_size = 0
        for param in model.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024**2
        
        return {
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'gflops': round(gflops, 2),
            'size_mb': round(size_mb, 2)
        }
    
    def _benchmark_speed(self, model, runs=100):
        """Benchmark inference speed"""
        
        model.model.eval()
        dummy_input = torch.randn(1, 3, self.imgsz, self.imgsz).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model.model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.time()
                _ = model.model(dummy_input)
                if self.device != 'cpu':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        fps = 1000.0 / mean_time
        
        return {
            'mean_time': round(mean_time, 2),
            'std_time': round(std_time, 2),
            'fps': round(fps, 2)
        }
    
    def compare_all(self, model_configs):
        """
        Compare all model variants
        
        Args:
            model_configs: List of tuples (model_path, model_name, weights_path)
        """
        
        for config in model_configs:
            if len(config) == 3:
                model_path, model_name, weights_path = config
            else:
                model_path, model_name = config
                weights_path = None
            
            self.evaluate_model(model_path, model_name, weights_path)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Sort by Top1 Accuracy
        df = df.sort_values('Top1_Accuracy', ascending=False)
        
        return df
    
    def save_results(self, df, output_path='comparison_results'):
        """Save comparison results"""
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = output_path / 'comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Save JSON
        json_path = output_path / 'comparison.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create visualizations
        self._plot_comparison(df, output_path)
        
        return df
    
    def _plot_comparison(self, df, output_path):
        """Create comparison plots"""
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('YOLOv8 Model Comparison: Baseline vs ECA vs SimAM vs Hybrid', 
                     fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax = axes[0, 0]
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df['Top1_Accuracy'], width, label='Top-1', alpha=0.8)
        ax.bar(x + width/2, df['Top5_Accuracy'], width, label='Top-5', alpha=0.8)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title('Classification Accuracy', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Parameters comparison
        ax = axes[0, 1]
        colors = sns.color_palette("husl", len(df))
        ax.bar(df['Model'], df['Parameters'] / 1e6, color=colors, alpha=0.8)
        ax.set_ylabel('Parameters (Millions)', fontweight='bold')
        ax.set_title('Model Parameters', fontweight='bold')
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 3. FLOPs comparison
        ax = axes[0, 2]
        ax.bar(df['Model'], df['GFLOPs'], color=colors, alpha=0.8)
        ax.set_ylabel('GFLOPs', fontweight='bold')
        ax.set_title('Computational Complexity', fontweight='bold')
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 4. Inference speed
        ax = axes[1, 0]
        ax.bar(df['Model'], df['Inference_Speed_ms'], color=colors, alpha=0.8)
        ax.set_ylabel('Time (ms)', fontweight='bold')
        ax.set_title('Inference Speed', fontweight='bold')
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 5. FPS
        ax = axes[1, 1]
        ax.bar(df['Model'], df['FPS'], color=colors, alpha=0.8)
        ax.set_ylabel('FPS', fontweight='bold')
        ax.set_title('Frames Per Second', fontweight='bold')
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 6. Efficiency (Accuracy / GFLOPs)
        ax = axes[1, 2]
        efficiency = df['Top1_Accuracy'] / (df['GFLOPs'] + 1e-6)  # Avoid division by zero
        ax.bar(df['Model'], efficiency, color=colors, alpha=0.8)
        ax.set_ylabel('Accuracy / GFLOPs', fontweight='bold')
        ax.set_title('Model Efficiency', fontweight='bold')
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'comparison_plots.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to: {output_path / 'comparison_plots.png'}")
        plt.close()


def main():
    """Example usage"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Compare YOLOv8 Classification Models')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data YAML file')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device ID')
    parser.add_argument('--output', type=str, default='comparison_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparison(
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device
    )
    
    # Define models to compare
    # Format: (model_config_path, display_name, weights_path)
    # If weights_path is None, model will be loaded from config
    model_configs = [
        ('ultralytics/cfg/models/v8/yolov8-cls.yaml', 'YOLOv8-Baseline', None),
        ('ultralytics/cfg/models/v8/yolov8_ECA_cls.yaml', 'YOLOv8-ECA', None),
        ('ultralytics/cfg/models/v8/yolov8_SimAM_cls.yaml', 'YOLOv8-SimAM', None),
        ('ultralytics/cfg/models/v8/yolov8_SimAM_backbone_cls.yaml', 'YOLOv8-SimAM-Backbone', None),
        ('ultralytics/cfg/models/v8/yolov8_SimAM_ECA_hybrid_cls.yaml', 'YOLOv8-Hybrid', None),
    ]
    
    # Run comparison
    print("\n" + "="*80)
    print("YOLOv8 Plant Leaf Disease Classification - Model Comparison")
    print("="*80)
    
    df = comparator.compare_all(model_configs)
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save results
    df = comparator.save_results(df, output_path=args.output)
    
    # Print best model
    best_model = df.iloc[0]
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"Model: {best_model['Model']}")
    print(f"Top-1 Accuracy: {best_model['Top1_Accuracy']:.4f}%")
    print(f"Top-5 Accuracy: {best_model['Top5_Accuracy']:.4f}%")
    print(f"Parameters: {best_model['Parameters']/1e6:.2f}M")
    print(f"GFLOPs: {best_model['GFLOPs']:.2f}")
    print(f"Inference Speed: {best_model['Inference_Speed_ms']:.2f} ms")
    print(f"FPS: {best_model['FPS']:.2f}")
    print("="*80)


if __name__ == '__main__':
    main()
