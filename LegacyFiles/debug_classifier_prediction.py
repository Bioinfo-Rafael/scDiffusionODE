#!/usr/bin/env python3
"""
Classifier予測精度チェック - CD19+ B細胞で何が起こっているか調査
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append(os.getcwd())

from guided_diffusion import dist_util
from guided_diffusion.script_util import create_classifier, classifier_and_diffusion_defaults

def check_classifier_prediction():
    """CD19+ B (type 3)でのclassifier予測を詳細チェック"""
    
    print("=== Classifier Prediction Analysis ===")
    
    # Setup
    dist_util.setup_dist()
    
    # Load classifier
    print("Loading classifier...")
    classifier_args = classifier_and_diffusion_defaults()
    classifier_args.update({
        'num_class': 11,
        'classifier_use_fp16': False
    })
    
    classifier = create_classifier(**classifier_args)
    classifier.load_state_dict(torch.load('output/checkpoint/classifier/pbmc68k_classifier/model009999.pt'))
    classifier.to(dist_util.dev())
    classifier.eval()
    
    # Generate some random samples to test classifier
    print("Testing classifier with random samples...")
    device = dist_util.dev()
    
    # Generate test samples (latent space dimension should match)
    batch_size = 100
    latent_dim = 128  # VAE latent dimension
    test_samples = torch.randn(batch_size, latent_dim, device=device)
    
    with torch.no_grad():
        # Get classifier predictions
        logits = classifier(test_samples, torch.zeros(batch_size, device=device))
        probs = F.softmax(logits, dim=-1)
        predicted_types = torch.argmax(probs, dim=1)
        
        print(f"\nPrediction results for {batch_size} random samples:")
        print(f"Logits shape: {logits.shape}")
        print(f"Probabilities shape: {probs.shape}")
        
        # Count predictions for each cell type
        unique_types, counts = torch.unique(predicted_types, return_counts=True)
        print(f"\nPredicted cell type distribution:")
        cell_type_names = [
            'CD8+ Cytotoxic T', 'CD8+/CD45RA+ Naive Cytotoxic', 'CD4+/CD45RO+ Memory',
            'CD19+ B', 'CD4+/CD25 T Reg', 'CD56+ NK', 'CD4+ T Helper2',
            'CD4+/CD45RA+/CD25- Naive T', 'CD34+', 'Dendritic', 'CD14+ Monocyte'
        ]
        
        for type_idx, count in zip(unique_types, counts):
            percentage = (count.item() / batch_size) * 100
            print(f"  Type {type_idx} ({cell_type_names[type_idx]}): {count.item()} ({percentage:.1f}%)")
        
        # Check if type 3 (CD19+ B) is ever predicted
        type_3_count = (predicted_types == 3).sum().item()
        print(f"\nCD19+ B (type 3) predictions: {type_3_count}/{batch_size} ({type_3_count/batch_size*100:.1f}%)")
        
        # Check max probabilities for type 3
        type_3_probs = probs[:, 3]
        print(f"CD19+ B probability stats:")
        print(f"  Mean: {type_3_probs.mean().item():.4f}")
        print(f"  Max: {type_3_probs.max().item():.4f}")
        print(f"  Min: {type_3_probs.min().item():.4f}")
        print(f"  Std: {type_3_probs.std().item():.4f}")
        
        # Check if any sample has high confidence for type 3
        high_conf_3 = (type_3_probs > 0.5).sum().item()
        print(f"  Samples with >50% confidence for CD19+ B: {high_conf_3}")
        
        # Overall prediction confidence
        max_probs = torch.max(probs, dim=1)[0]
        print(f"\nOverall prediction confidence:")
        print(f"  Mean max probability: {max_probs.mean().item():.4f}")
        print(f"  Samples with >90% confidence: {(max_probs > 0.9).sum().item()}")
        print(f"  Samples with >50% confidence: {(max_probs > 0.5).sum().item()}")

if __name__ == "__main__":
    check_classifier_prediction()
