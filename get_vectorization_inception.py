import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, directory, cut):
        self.directory = directory
        self.cut_files = cut
        self.file_paths = self._load_file_paths()
        
    def _load_file_paths(self):
        files = [f for f in os.listdir(self.directory) if f.endswith('.npz')][:self.cut_files]
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return [os.path.join(self.directory, f) for f in files]
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        images = data['images'].reshape(-1, 50, 50)  # Shape: (660, 50, 50)

        images = torch.from_numpy(images).float()
        images = images.unsqueeze(1)  # Shape: (660, 1, 50, 50)
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return images, filename

def create_dataloader(directory, cut, batch_size=1, num_workers=0):
    dataset = CustomDataset(directory, cut)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

class InceptionVectorizer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._prepare_model().to(self.device).eval()
        
    def _prepare_model(self):
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        model.aux_logits = False
        model.AuxLogits = None
        model.fc = torch.nn.Identity()
        return model
    
    def _preprocess(self, x):
        # Resize to 299x299 for InceptionV3

        x_resized = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Convert from [0, 255] to ImageNet normalized range
        x_resized = x_resized / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x_resized - mean) / std
    
    def vectorize_batch(self, batch):
        """
        Process batch of files
        Input shape: (B, 660, 1, 50, 50)
        Output shape: (B, 2048)
        """
        #print(batch.shape)
        B, N, C, H, W = batch.shape
        
        # Group into triplets (660 images = 220 triplets)
        triplets = batch.view(B, 220, 3, C, H, W)  # (B, 220, 3, 1, 50, 50)
        
        # Combine triplet and channel dimensions
        triplets = triplets.permute(0, 1, 3, 2, 4, 5)  # (B, 220, 1, 3, 50, 50)
        triplets = triplets.reshape(-1, 3, H, W)       # (B*220, 3, 50, 50)
        
        # Preprocess and get features
        with torch.no_grad():
            processed = self._preprocess(triplets.to(self.device))
            features = self.model(processed)  # (B*220, 2048)
            
        # Average features per file
        features = features.view(B, 220, -1)  # (B, 220, 2048)
        return torch.mean(features, dim=1)     # (B, 2048)

image_dir = './data/validation'
validation_dir = './data/validation/'
dataloader = create_dataloader(validation_dir, cut=2905, batch_size=4)

vectorizer = InceptionVectorizer()

all_vectors = []
all_filenames = []

with tqdm(total=len(dataloader), desc="Processing files", unit="batch") as pbar:
    for batch_images, batch_filenames in dataloader:
        batch_images = batch_images.to(vectorizer.device)
        vectors = vectorizer.vectorize_batch(batch_images)
        
        all_vectors.append(vectors.cpu().numpy())
        all_filenames.extend(batch_filenames)

        pbar.set_postfix({
            'current_files': batch_filenames,
            'sets_of_vectors_generated': len(all_vectors)
        })
        pbar.update(1)

final_vectors = np.concatenate(all_vectors, axis=0)

path = "./data/vectorization/try.npz"
np.savez(path, final_vectors)