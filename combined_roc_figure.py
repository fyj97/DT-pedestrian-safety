#!/usr/bin/env python3
"""
Script to generate side-by-side ROC curves for LaTeX figure.
Generates both TTC threshold and danger distance threshold ROC curves.
"""

import numpy as np
import matplotlib.pyplot as plt

def calculate_tpr_fpr(confusion_matrix):
    """Calculate True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity)"""
    TP, FP = confusion_matrix[0]
    FN, TN = confusion_matrix[1]
    
    # Avoid division by zero
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    return TPR, FPR

def plot_combined_roc_figure():
    """Create side-by-side ROC curves for LaTeX figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # Load TTC data
    try:
        ttc_confusion_matrices = np.load('ttc_threshold_confusion_matrices.npy')
        print(f"Loaded TTC confusion matrices with shape: {ttc_confusion_matrices.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'ttc_threshold_confusion_matrices.npy'")
        print("Please run the TTC analysis first to generate the data.")
        return
    
    # Load danger distance data
    try:
        dd_confusion_matrices = np.load('dd_confusion_matrices.npy')
        print(f"Loaded danger distance confusion matrices with shape: {dd_confusion_matrices.shape}")
    except FileNotFoundError:
        print("Error: Could not find 'dd_confusion_matrices.npy'")
        print("Please run the danger distance analysis first to generate the data.")
        return
    
    # Calculate TTC ROC curve
    ttc_threshold_values = list(range(1, 13))
    ttc_times = [t * 0.1 for t in ttc_threshold_values]  # Convert to seconds
    ttc_tprs = []
    ttc_fprs = []
    
    for cm in ttc_confusion_matrices:
        tpr, fpr = calculate_tpr_fpr(cm)
        ttc_tprs.append(tpr)
        ttc_fprs.append(fpr)
    
    # Calculate danger distance ROC curve
    dd_threshold_values = list(range(10, 201, 5))
    dd_tprs = []
    dd_fprs = []
    
    for cm in dd_confusion_matrices:
        tpr, fpr = calculate_tpr_fpr(cm)
        dd_tprs.append(tpr)
        dd_fprs.append(fpr)
    
    # Plot TTC ROC curve (left subplot)
    # Add (0,0) and (1,1) points to connect the curve
    ttc_extended_fprs = [0] + ttc_fprs + [1]
    ttc_extended_tprs = [0] + ttc_tprs + [1]
    
    ax1.plot(ttc_extended_fprs, ttc_extended_tprs, 'b-o', linewidth=2, markersize=6, label='ROC Curve')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=18)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=18)
    ax1.legend(loc="lower right", fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Varying TTC threshold (in seconds)', fontsize=16, fontweight='bold')
    
    # Mark the 1.1 second point (TTC threshold 11)
    target_time = 1.1
    target_idx = None
    
    for i, ttc in enumerate(ttc_times):
        if abs(ttc - target_time) < 0.01:
            target_idx = i
            break
    
    if target_idx is not None:
        target_fpr = ttc_fprs[target_idx]
        target_tpr = ttc_tprs[target_idx]
        
        ax1.annotate(f'Selected Point\nTTC={target_time}s\nTPR={target_tpr:.3f}, FPR={target_fpr:.3f}', 
                    (target_fpr, target_tpr), 
                    xytext=(-150, -50), textcoords='offset points',
                    fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='green', lw=2))
        
        ax1.plot(target_fpr, target_tpr, 'gs', markersize=12, label=f'Selected Point (TTC={target_time}s)')
    
    # Plot danger distance ROC curve (right subplot)
    # Add (0,0) and (1,1) points to connect the curve
    dd_extended_fprs = [0] + dd_fprs + [1]
    dd_extended_tprs = [0] + dd_tprs + [1]
    
    ax2.plot(dd_extended_fprs, dd_extended_tprs, 'r-o', linewidth=2, markersize=6, label='ROC Curve')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=18)
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=18)
    ax2.legend(loc="lower right", fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Varying danger distance threshold (in pixels)', fontsize=16, fontweight='bold')
    
    # Mark the selected point (threshold 30)
    target_threshold = 30
    target_idx = None
    
    for i, threshold in enumerate(dd_threshold_values):
        if threshold == target_threshold:
            target_idx = i
            break
    
    if target_idx is not None:
        target_fpr = dd_fprs[target_idx]
        target_tpr = dd_tprs[target_idx]
        
        ax2.annotate(f'Selected Point\nThreshold={target_threshold}\nTPR={target_tpr:.3f}, FPR={target_fpr:.3f}', 
                    (target_fpr, target_tpr), 
                    xytext=(-10, -80), textcoords='offset points',
                    fontsize=14, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='green', lw=2))
        
        ax2.plot(target_fpr, target_tpr, 'gs', markersize=12, label=f'Selected Point (Threshold={target_threshold})')
    
    plt.tight_layout()
    plt.savefig('combined_roc_figure.png', dpi=400, bbox_inches='tight')
    plt.show()
    
    print("Combined ROC figure saved as 'combined_roc_figure.png'")
    
    # Calculate and print AUC scores
    # TTC AUC
    ttc_sorted_data = sorted(zip(ttc_fprs, ttc_tprs, ttc_threshold_values))
    ttc_sorted_fprs = [x[0] for x in ttc_sorted_data]
    ttc_sorted_tprs = [x[1] for x in ttc_sorted_data]
    
    if ttc_sorted_fprs[0] != 0:
        ttc_sorted_fprs.insert(0, 0)
        ttc_sorted_tprs.insert(0, 0)
    if ttc_sorted_fprs[-1] != 1:
        ttc_sorted_fprs.append(1)
        ttc_sorted_tprs.append(1)
    
    ttc_auc_score = np.trapz(ttc_sorted_tprs, ttc_sorted_fprs)
    
    # Danger distance AUC
    dd_sorted_data = sorted(zip(dd_fprs, dd_tprs, dd_threshold_values))
    dd_sorted_fprs = [x[0] for x in dd_sorted_data]
    dd_sorted_tprs = [x[1] for x in dd_sorted_data]
    
    if dd_sorted_fprs[0] != 0:
        dd_sorted_fprs.insert(0, 0)
        dd_sorted_tprs.insert(0, 0)
    if dd_sorted_fprs[-1] != 1:
        dd_sorted_fprs.append(1)
        dd_sorted_tprs.append(1)
    
    dd_auc_score = np.trapz(dd_sorted_tprs, dd_sorted_fprs)
    
    print(f"\nAUC Scores:")
    print(f"TTC ROC AUC: {ttc_auc_score:.3f}")
    print(f"Danger Distance ROC AUC: {dd_auc_score:.3f}")

def main():
    """Generate the combined ROC figure"""
    plot_combined_roc_figure()

if __name__ == "__main__":
    main() 