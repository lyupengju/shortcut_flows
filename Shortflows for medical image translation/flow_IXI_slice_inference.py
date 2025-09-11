import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 

import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# MONAI imports
import monai
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, ToTensord, EnsureTyped, Resized, ScaleIntensityd,
    CopyItemsd, Lambdad, CropForegroundd, SpatialPadd, ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd, CenterSpatialCropd, Invertd, SaveImaged
)
from monai.utils import first, set_determinism
from flow_Unet import DiffusionModelUNet
from meanflow import MeanFlow

# Set determinism for reproducible results
set_determinism(seed=50)

class AxialSliceProcessor:
    """处理3D数据axial切片的工具类"""
    
    def __init__(self, out_path):
        """
        初始化axial切片处理器
        
        Args:
            out_path: 输出路径
        """
        self.out_path = out_path
        
    def get_3d_preprocessing_transforms(self):
        """获取3D预处理变换"""
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0.5, upper=99.5, b_min=0, b_max=1,clip = True),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, -1)),
        ])
        return transforms
    
    def extract_specific_slice(self, preprocessed_data, slice_idx):
        """
        从预处理后的3D数据中提取特定的axial切片
        
        Args:
            preprocessed_data: 包含预处理后image和label的字典
            slice_idx: 要提取的切片索引
        
        Returns:
            slice_data: 包含切片数据和元数据的字典
        """
        image_3d = preprocessed_data["image"]  # Shape: (1, H, W, D)
        label_3d = preprocessed_data["label"]  # Shape: (1, H, W, D)
        
        # 获取axial维度（假设是最后一个维度）
        axial_depth = image_3d.shape[-1]
        print(f"Original axial depth: {axial_depth}")
        
        # 检查切片索引是否有效
        if slice_idx < 0 or slice_idx >= axial_depth:
            raise ValueError(f"Slice index {slice_idx} is out of range [0, {axial_depth-1}]")
        
        # 提取指定的axial切片
        image_slice = image_3d[..., slice_idx]  # Shape: (1, H, W)
        label_slice = label_3d[..., slice_idx]  # Shape: (1, H, W)
        
        slice_data = {
            "image_slice": image_slice,
            "label_slice": label_slice,
            "slice_idx": slice_idx,
            "original_shape": image_3d.shape,
            "axial_depth": axial_depth,
            "slice_position": slice_idx / (axial_depth - 1) if axial_depth > 1 else 0.0
        }
        
        # 保存原始3D数据的元信息
        if "image_meta_dict" in preprocessed_data:
            slice_data["original_meta"] = preprocessed_data["image_meta_dict"]
        
        print(f"Extracted axial slice at index {slice_idx}")
        return slice_data

class MRITranslationInferencer:
    def __init__(self, model_path, out_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize axial slice processor
        self.slice_processor = AxialSliceProcessor(out_path)
        
        # Initialize model and load weights
        self.model = self._create_model()
        self.model.to(self.device)
        self._load_model_weights(model_path)
        self.meanflow = MeanFlow(flow_ratio=0.50)


        
    def _create_model(self):
        """Create 2D U-Net model for slice-wise translation"""
        model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(64, 64, 64),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=64,
            with_conditioning=False,
        )
        return model
    
    def _load_model_weights(self, model_path):
        """加载训练好的模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model loaded successfully!")
    
    def preprocess_single_volume(self, image_path, label_path, slice_idx):
        """预处理单个3D体积并提取特定切片"""
        print(f"Preprocessing volume: {image_path}")
        
        # 获取预处理变换
        preprocess_3d = self.slice_processor.get_3d_preprocessing_transforms()
        
        # 应用预处理
        data_dict = {"image": str(image_path), "label": str(label_path)}
        preprocessed_data = preprocess_3d(data_dict)
        
        # 提取特定切片
        slice_data = self.slice_processor.extract_specific_slice(
            preprocessed_data, slice_idx
        )
        
        return slice_data
    
    def inference_single_slice(self, image_slice):
        """对单个2D切片进行推理"""
        self.model.eval()
        image_slice = image_slice.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.autocast("cuda"):
        
                current_img = self.meanflow.sample_onestep(image_slice,self.model)
                # current_img = self.meanflow.sample_multistep(image_slice,self.model,sample_steps=1)
        return current_img.squeeze(0)  # Remove batch dimension
        # Save intermediate results as images

        # output_dir = os.path.join(self.slice_processor.out_path, "intermediate_results")
        # os.makedirs(output_dir, exist_ok=True)
              
        # saved_paths = []
        # for i, img in enumerate(current_img):
        #     # Remove batch dimension and convert to numpy
        #     img_np = img.squeeze(0).cpu().numpy()[0] # Shape: (H, W)
            
        #     # Save as image
        #     plt.figure(figsize=(6, 6))
        #     plt.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        #     plt.axis('off')
        #     img_path = os.path.join(output_dir, f"intermediate_step_{i:03d}.png")
        #     plt.savefig(img_path, dpi=150, bbox_inches='tight', pad_inches=0)
        #     plt.close()
        #     saved_paths.append(img_path)
    
        # print(f"Saved {len(saved_paths)} intermediate results to {output_dir}") 
        
        # return current_img[-1].squeeze(0)  # Remove batch dimension

       
    
    def save_slice_comparison(self, slice_data, predicted_slice, output_dir, base_filename):
        """保存切片对比图和单独的切片"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取数据
        image_slice = slice_data["image_slice"].cpu().numpy()[0]  # Remove channel dimension
        label_slice = slice_data["label_slice"].cpu().numpy()[0]  # Remove channel dimension
        predicted_slice_np = predicted_slice.cpu().numpy()[0]  # Remove channel dimension
        slice_idx = slice_data["slice_idx"]
        
        # 1. 保存对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 输入切片 (Image)
        axes[0].imshow(image_slice, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Input Image (Slice {slice_idx})')
        axes[0].axis('off')
        
        # 标签切片 (Label)
        axes[1].imshow(label_slice, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Ground Truth Label (Slice {slice_idx})')
        axes[1].axis('off')
        
        # 预测切片 (Prediction)
        axes[2].imshow(predicted_slice_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Predicted (Slice {slice_idx})')
        axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, f"{base_filename}_slice_{slice_idx:03d}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison saved to: {comparison_path}")
        
        # 2. 保存单独的切片图像
        # Image slice
        plt.figure(figsize=(6, 6))
        plt.imshow(image_slice, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        image_path = os.path.join(output_dir, f"{base_filename}_slice_{slice_idx:03d}_image.png")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Label slice
        plt.figure(figsize=(6, 6))
        plt.imshow(label_slice, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        label_path = os.path.join(output_dir, f"{base_filename}_slice_{slice_idx:03d}_label.png")
        plt.savefig(label_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Prediction slice
        plt.figure(figsize=(6, 6))
        plt.imshow(predicted_slice_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        prediction_path = os.path.join(output_dir, f"{base_filename}_slice_{slice_idx:03d}_prediction.png")
        plt.savefig(prediction_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Individual slices saved:")
        print(f"  Image: {image_path}")
        print(f"  Label: {label_path}")
        print(f"  Prediction: {prediction_path}")
        
        # # 3. 保存为numpy数组
        # np_output_dir = os.path.join(output_dir, "numpy_arrays")
        # os.makedirs(np_output_dir, exist_ok=True)
        
        # np.save(os.path.join(np_output_dir, f"{base_filename}_slice_{slice_idx:03d}_image.npy"), image_slice)
        # np.save(os.path.join(np_output_dir, f"{base_filename}_slice_{slice_idx:03d}_label.npy"), label_slice)
        # np.save(os.path.join(np_output_dir, f"{base_filename}_slice_{slice_idx:03d}_prediction.npy"), predicted_slice_np)
        
        return {
            "comparison_path": comparison_path,
            "image_path": image_path,
            "label_path": label_path,
            "prediction_path": prediction_path
        }
    
    def inference_specific_slice(self, image_path, label_path, slice_idx, output_dir):
        """对特定切片进行推理"""
        print(f"\n=== Starting inference for slice {slice_idx} of {image_path} ===")
        
        # 预处理并提取特定切片
        slice_data = self.preprocess_single_volume(image_path, label_path, slice_idx)
        
        # 对切片进行推理
        image_slice = slice_data["image_slice"]
        print("Performing slice inference...")
        predicted_slice = self.inference_single_slice(image_slice)
        
        # 保存结果
        base_filename = Path(image_path).stem
        saved_paths = self.save_slice_comparison(slice_data, predicted_slice, output_dir, base_filename)
        
        print(f"Inference completed for slice {slice_idx}!")
        return slice_data, predicted_slice, saved_paths
    
    def inference_multiple_slices(self, image_path, label_path, slice_indices, output_dir):
        """对多个指定切片进行推理"""
        print(f"\n=== Starting inference for multiple slices {slice_indices} of {image_path} ===")
        
        results = {}
        base_filename = Path(image_path).stem
        
        for slice_idx in slice_indices:
            print(f"\nProcessing slice {slice_idx}...")
            try:
                slice_data, predicted_slice, saved_paths = self.inference_specific_slice(
                    image_path, label_path, slice_idx, output_dir
                )
                results[slice_idx] = {
                    "slice_data": slice_data,
                    "predicted_slice": predicted_slice,
                    "saved_paths": saved_paths
                }
            except Exception as e:
                print(f"Error processing slice {slice_idx}: {e}")
                results[slice_idx] = {"error": str(e)}
        
        print(f"\nCompleted inference for {len(results)} slices!")
        return results


def main():
    parser = argparse.ArgumentParser(description='MRI Translation Inference for Specific Slices')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--label_path', type=str, required=True,
                        help='Path to input label')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Specific slice index to process')
    parser.add_argument('--slice_indices', type=str, default=None,
                        help='Multiple slice indices separated by comma (e.g., "10,20,30")')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = MRITranslationInferencer(args.model_path, args.output_path)
    
    # 处理切片索引
    if args.slice_indices is not None:
        # 处理多个切片
        slice_indices = [int(idx.strip()) for idx in args.slice_indices.split(',')]
        results = inferencer.inference_multiple_slices(
            args.image_path, 
            args.label_path, 
            slice_indices, 
            args.output_path
        )
    elif args.slice_idx is not None:
        # 处理单个切片
        slice_data, predicted_slice, saved_paths = inferencer.inference_specific_slice(
            args.image_path, 
            args.label_path, 
            args.slice_idx, 
            args.output_path
        )
    else:
        print("Please specify either --slice_idx or --slice_indices")
        return


if __name__ == "__main__":
    # 示例用法
    if len(os.sys.argv) == 1:
        # 如果没有命令行参数，使用默认配置运行示例
        
        model_path = "flow/IXI/model_epoch_28.pth"  # 训练好的模型路径
        # image_path = "/home/plyu/data/IXI/IXI-T2/IXI013-HH-1212-T2.nii.gz"  # 输入图像路径
        # label_path = "/home/plyu/data/IXI/IXI-PD/IXI013-HH-1212-PD.nii.gz"
        # output_path = "IXI_flow_output_57"  # 输出目录

        #reverse
        label_path = "/home/data/IXI/IXI-T2/IXI013-HH-1212-T2.nii.gz"  # 输入图像路径
        image_path  = "/home/data/IXI/IXI-PD/IXI013-HH-1212-PD.nii.gz"
        output_path = "IXI_flow_output_57_reverse"  # 输出目录
        
        # 初始化推理器
        inferencer = MRITranslationInferencer(model_path, output_path)
        
        # 示例1: 推理特定切片 (例如第50层)
        slice_data, predicted_slice, saved_paths = inferencer.inference_specific_slice(
            image_path, label_path, 57, output_path
        )
        
        # # 示例2: 推理多个切片
        # results = inferencer.inference_multiple_slices(
        #     image_path, label_path, [40, 50, 60, 70], output_path
        # )
    else:
        main()