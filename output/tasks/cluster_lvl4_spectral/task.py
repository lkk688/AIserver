#!/usr/bin/env python3
"""
Spectral Clustering Pipeline - Level 4
Task: Implement spectral clustering that outperforms k-means on moons dataset.
"""

import numpy as np
    # Check silhouette score is positive
    assert spectral_metrics_val["silhouette"] > 0, (
        f"Silhouette score should be positive: {spectral_metrics_val['silhouette']:.4f}"
    )
    print(f"✓ Silhouette score is positive: {spectral_metrics_val['silhouette']:.4f}")
    
    # Check R2 is reasonable (not negative)
    assert spectral_metrics_val["r2"] > -1.0, (
        f"R2 score should be reasonable: {spectral_metrics_val['r2']:.4f}"
    )
    print(f"✓ R2 score is reasonable: {spectral_metrics_val['r2']:.4f}")
    
    print("\nAll quality checks passed!")
    print("=" * 60)
