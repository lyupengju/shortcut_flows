import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # 

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# MONAI imports
from monai.data import Dataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, RandRotated,
    ToTensord, EnsureTyped, RandGaussianNoised, Resized, ScaleIntensityd,
    CopyItemsd, Lambdad, CropForegroundd, SpatialPadd, ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd, CenterSpatialCropd,SaveImaged
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import SSIMLoss
from monai.metrics import SSIMMetric
from monai.utils import first, set_determinism
from flow_Unet import DiffusionModelUNet
from meanflow import MeanFlow

# Set determinism for reproducible results
set_determinism(seed=50)

class AxialSliceProcessor:
    """处理3D数据axial切片的工具类"""
    
    def __init__(self, margin,):
       
        self.middle_percent = margin #(10%)
        
    def get_3d_preprocessing_transforms(self):
        """获取3D预处理变换"""
        transforms = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            # SpatialPadd(keys=["image", "label"], spatial_size=(144, 144, -1), method='end', mode='constant'),
            ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0.5, upper=99.5, b_min=0, b_max=1,clip = True),
            
            
            CenterSpatialCropd(keys=["image", "label"], roi_size=(256, 256, -1)), #ixi



            # SaveImaged(keys="image", meta_keys="image_meta_dict", output_dir='output',
            #                    output_postfix="image", output_ext=".nii.gz", resample=False),
            # SaveImaged(keys="labed", meta_keys="labed_meta_dict", output_dir='output',
            #                    output_postfix="label", output_ext=".nii.gz", resample=False)
        ])
        return transforms
    
    def extract_axial_slices_with_metadata(self, preprocessed_data, use_middle_80_percent=True):
        """
        从预处理后的3D数据中提取axial切片
        
        Args:
            preprocessed_data: 包含预处理后image和label的字典
            use_middle_80_percent: 是否只使用中间80%的切片（训练时）
        
        Returns:
            slice_data_list: 包含切片数据和元数据的列表
        """
        image_3d = preprocessed_data["image"]  # Shape: (1, H, W, D)
        print('hha',image_3d.shape)
        label_3d = preprocessed_data["label"]  # Shape: (1, H, W, D)
        print('hha',image_3d.shape)

        # 获取axial维度（假设是最后一个维度）
        axial_depth = image_3d.shape[-1]
        print(f"Original axial depth: {axial_depth}")
        
        # 确定切片范围
        if use_middle_80_percent:
            # 训练时使用中间80%
            margin = int(axial_depth * self.middle_percent)  # 10% margin on each side
            start_idx = margin
            end_idx = axial_depth - margin
            print(f"Using middle 80% slices: {start_idx} to {end_idx-1}")
        else:
            # 推理时使用所有切片
            start_idx = 0
            end_idx = axial_depth
            print(f"Using all slices: {start_idx} to {end_idx-1}")
        
        slice_data_list = []
        
        for i in range(start_idx, end_idx):
            # 提取单个axial切片
            image_slice = image_3d[..., i]  # Shape: (1, H, W)
            label_slice = label_3d[..., i]  # Shape: (1, H, W)
            
            slice_data = {
                "image_slice": image_slice,
                "label_slice": label_slice,
                "slice_idx": i,
                "original_shape": image_3d.shape,
                "axial_depth": axial_depth,
                "slice_position": i / (axial_depth - 1) if axial_depth > 1 else 0.0
            }
            
            # 保存原始3D数据的元信息（用于推理时重建）
            if "image_meta_dict" in preprocessed_data:
                slice_data["original_meta"] = preprocessed_data["image_meta_dict"]
            
            slice_data_list.append(slice_data)
        
        print(f"Extracted {len(slice_data_list)} axial slices")
        return slice_data_list

class MRITranslationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize axial slice processor
        self.slice_processor = AxialSliceProcessor(
            margin=self.config['margin'],
        )
        self.snapshot_path = self.config['snapshot_path']
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        # self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=3e-5)
        self.scaler = torch.GradScaler("cuda")
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
    
    def _get_2d_slice_transforms(self, is_training=True):
        """定义2D切片的数据变换"""
        if is_training:
            transforms = Compose([
                # # 2D数据增强
                # RandFlipd(keys=["image_slice", "label_slice"], prob=0.3, spatial_axis=[0, 1]),
                # RandRotated(keys=["image_slice", "label_slice"], range_x=0.2, range_y=0.2, prob=0.3),
                # RandGaussianNoised(keys=["image_slice"], prob=0.1, mean=0.0, std=0.01),
                ToTensord(keys=["image_slice", "label_slice"]),
            ])
        else:
            transforms = Compose([
                ToTensord(keys=["image_slice", "label_slice"]),
            ])
        return transforms
    
    def prepare_data(self, data_dir):
        """准备训练和验证数据集 - 将3D数据转换为2D切片"""
   
        # 获取3D预处理变换
        preprocess_3d = self.slice_processor.get_3d_preprocessing_transforms()
        
        # 创建2D切片数据字典
        print("Processing training data...")
        train_slice_dicts = self._create_slice_data_dicts(data_dir, preprocess_3d, use_middle_80_percent=True,if_train = True)
        val_slice_dicts = train_slice_dicts[:100]
        
        
        
        print(f"训练切片数量: {len(train_slice_dicts)}")
        print(f"验证切片数量: {len(val_slice_dicts)}")
        
        # 创建2D切片变换
        train_transforms = self._get_2d_slice_transforms(is_training=True)
        val_transforms = self._get_2d_slice_transforms(is_training=False)
        
        # 创建数据集
        train_ds = Dataset(data=train_slice_dicts, transform=train_transforms)
        val_ds = Dataset(data=val_slice_dicts, transform=val_transforms)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4
        )
        
        return train_loader, val_loader
    
    def _create_slice_data_dicts(self, data_dir, preprocess_3d, use_middle_80_percent=True,if_train = True):
        """从3D数据创建2D切片数据字典"""
        slice_data_dicts = []
        if if_train == True:
            
            image_dir = Path(data_dir) / "IXI-T2"
            label_dir = Path(data_dir) / "IXI-PD"

            image_files = sorted([f for f in image_dir.glob("*-T2.nii.gz")])

            for img_file in image_files:
                case_id = img_file.name.replace("-T2.nii.gz", "")
                label_file = label_dir / f"{case_id}-PD.nii.gz"


                if not label_file.exists():
                    print(f"⚠️ 找不到标签文件: {label_file.name}，跳过。")
                    continue

        
                try:
                    # 应用3D预处理
                    data_dict = {"image": str(img_file), "label": str(label_file)}
                    # print('sdfa',data_dict['image'])
                    preprocessed_data = preprocess_3d(data_dict)
                    
                    # 提取axial切片
                    slice_list = self.slice_processor.extract_axial_slices_with_metadata(
                        preprocessed_data, use_middle_80_percent
                    )
                    
                    # 为每个切片创建数据字典
                    for slice_data in slice_list:
                        slice_dict = {
                            "image_slice": slice_data["image_slice"],
                            "label_slice": slice_data["label_slice"],
                            "subject_id": str(img_file),
                            "slice_idx": slice_data["slice_idx"],
                            "slice_position": slice_data["slice_position"],
                            "original_shape": slice_data["original_shape"]
                        }
                        
                        if "original_meta" in slice_data:
                            slice_dict["original_meta"] = slice_data["original_meta"]
                        
                        slice_data_dicts.append(slice_dict)
                        
                except Exception as e:
                    print(f"Error processing {img_file.stem}: {e}")
                    continue
            print('**************data preprocessing finished*********')
            print(len(slice_data_dicts))
     
        return slice_data_dicts
    
    
   
    def train_epoch(self, train_loader,val_loader, n_epochs):

        val_interval = 2
        for epoch in range(n_epochs):
            self.model.train()
            train_epoch_loss = 0
            epoch_iterator_train = tqdm(train_loader, desc="Training", dynamic_ncols=True)

            for step, data in enumerate(epoch_iterator_train):
                images = data["image_slice"].cuda()
                seg = data["label_slice"].cuda()  # this is the ground truth segmentation
                # print('prior', torch.unique(seg))
                # print(torch.unique(seg))

                self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast("cuda"):

                    loss, mse_val = self.meanflow.loss(self.model, images, seg)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                epoch_iterator_train.set_description( "Training loss=%2.5f)" % (loss))

                train_epoch_loss += loss.item()
            train_loss_per_epoch =   train_epoch_loss / (step + 1)

            # writer.add_scalar('loss/train_loss_per_epoch', train_loss_per_epoch, epoch)# save to tensorboard 保存每个epoch的loss

            if epoch % val_interval == 0:
                self.model.eval()
                val_epoch_loss = 0
                for step, data_val in enumerate(val_loader): 
                    images = data_val["image_slice"].cuda()
                    seg = data_val["label_slice"].cuda()  # this is the ground truth segmentation
                    with torch.no_grad():
                        with torch.autocast("cuda"):
                         
                            pred = self.meanflow.sample_onestep(images,self.model)
                            val_loss = F.mse_loss(pred.float(), seg.float())
                            val_epoch_loss += val_loss.item()

                val_loss_per_epoch =   val_epoch_loss / (step + 1)
            
                print("Epoch", epoch, "Validation loss", val_loss_per_epoch)
                # writer.add_scalar('loss/eval_loss', val_epoch_loss / (step + 1), epoch)# save to tensorboard 保存每个epoch的loss
                print('Save the latest best model')
                save_mode_path = os.path.join(self.snapshot_path, "model_epoch_"+ str(epoch)  + '.pth')
                torch.save(self.model.state_dict(), save_mode_path) #save model
        
        save_final_path = os.path.join(self.snapshot_path, 'segmodel.pth')
        torch.save(self.model.state_dict(), save_final_path)
        print(f"train diffusion completed.")

    
 
def main():
    # 配置参数
    config = {
        'margin': 0.2,
        'batch_size': 6,
        'model_path': 'flow/BraTS/model_epoch_19.pth',
        'num_epochs': 100,
        'snapshot_path' : 'flow/IXI'
    }
    
    # 初始化训练器
    trainer = MRITranslationTrainer(config)
    
    # 准备数据
    data_dir = "/home/data/IXI/"
    
    
    print("开始准备数据...")
    train_loader, val_loader = trainer.prepare_data(data_dir)
    
    # 训练模型
    print("开始训练...")
    trainer.train_epoch( train_loader,val_loader,config['num_epochs'])
    
    # 可视化结果


if __name__ == "__main__":
    main()