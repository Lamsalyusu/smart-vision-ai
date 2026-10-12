# backend/ml_model/depth_estimation.py
import numpy as np
import cv2
import torch
import os

class DepthEstimator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def _initialize(self):
        if self._initialized:
            return
            
        print(" Loading MiDaS depth estimation mode")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Using device: {self.device}")
        
        # Load MiDaS Small (fastest, ~25MB)
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
        self.transform = midas_transforms.small_transform
        
        self._initialized = True
        print(" MiDaS loaded successfully!")
    
    def estimate_depth(self, image_bytes):
        self._initialize()
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform and run inference
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original image dimensions
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-1 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)
        
        return depth_map, img_rgb
    
    def generate_point_cloud(self, image_bytes, max_points=16384, depth_scale=2.0):
        depth_map, img_rgb = self.estimate_depth(image_bytes)
        
        h, w = depth_map.shape

        total_pixels = h * w
        step = max(1, int(np.sqrt(total_pixels / max_points)))
        
        positions = []
        colors = []
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                depth = float(depth_map[y, x])
                
                # Skip very low depth (background noise)
                if depth < 0.05:
                    continue
                
                px = (x / w - 0.5) * 2.0
                py = -(y / h - 0.5) * 2.0
                pz = depth * depth_scale
                
                positions.extend([px, py, pz])
                
                # Normalize color to 0-1
                r, g, b = img_rgb[y, x]
                colors.extend([r / 255.0, g / 255.0, b / 255.0])
        
        point_count = len(positions) // 3
        
        return {
            'positions': positions,
            'colors': colors,
            'point_count': point_count,
            'original_width': w,
            'original_height': h,
            'step': step
        }

depth_estimator = DepthEstimator()