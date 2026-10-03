"""
DermaScope Model Architecture with EfficientNet-B4 backbone and FiLM metadata fusion.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Modulates visual features based on metadata.
    """
    def __init__(self, metadata_dim: int, feature_dim: int):
        super(FiLMLayer, self).__init__()
        # Small MLP to encode metadata into gamma and beta vectors
        self.mlp = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim * 2) # Output gamma and beta
        )
        
    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        film_params = self.mlp(metadata)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Broadcast gamma and beta across spatial dimensions if features are 4D (N, C, H, W)
        if features.dim() == 4:
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
            
        # Modulate: output = gamma * features + beta
        return features * gamma + beta

class DermaScope(nn.Module):
    """
    Multimodal deep learning system for early melanoma detection.
    Uses EfficientNet-B4 backbone and FiLM for metadata fusion.
    """
    def __init__(self, num_classes: int = 7, metadata_dim: int = 18, backbone_name: str = 'efficientnet_b4'):
        super(DermaScope, self).__init__()
        
        # Backbone (EfficientNet-B4 by default)
        if backbone_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
            feature_dim = 1792
        else:
            self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            feature_dim = 1536
            
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # FiLM Layer for metadata conditioning
        self.film = FiLMLayer(metadata_dim=metadata_dim, feature_dim=feature_dim)
        
        # Custom classification head with MC Dropout for uncertainty
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # Extract visual features (N, C, H, W)
        features = self.backbone.features(image)
        
        # Apply GAP -> (N, C, 1, 1) -> (N, C)
        features = self.gap(features).flatten(1)
        
        # Modulate features with metadata using FiLM
        fused_features = self.film(features, metadata)
        
        # Classification
        out = self.head(fused_features)
        return out
        
    def freeze_backbone(self):
        """Freezes the EfficientNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, num_layers: int = None):
        """
        Unfreezes the backbone. If num_layers is specified, only unfreezes the last N layers.
        """
        if num_layers is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze only the last num_layers
            layers = list(self.backbone.features.children())
            for layer in layers[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def predict_with_uncertainty(self, image: torch.Tensor, metadata: torch.Tensor, num_samples: int = 30):
        """
        Monte Carlo Dropout for uncertainty estimation.
        """
        self.train() # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(image, metadata)
                probs = torch.softmax(out, dim=1)
                predictions.append(probs.unsqueeze(0))
                
        predictions = torch.cat(predictions, dim=0) # (num_samples, batch, num_classes)
        mean_probs = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0).mean(dim=1) # simplified uncertainty metric
        return mean_probs, uncertainty
