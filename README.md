
# Cancer Detection & Cell Type Classification Using Semi-Supervised CNNs

## Introduction & Problem Context

Accurate classification of cancerous cells in histopathological images is a high-stakes task with direct clinical impact. However, the domain is constrained by several challenges typical of medical imaging:

- **Limited annotated data:** Labeling requires expert pathologists, making datasets small and expensive to create.
- **Patient heterogeneity:** Variations in tissue morphology and staining across patients introduce significant domain shifts.
- **Class imbalance:** Cancerous regions are often underrepresented compared to normal tissue.

Conventional fully supervised deep learning models typically overfit due to limited labeled data and fail to generalize well across patients, restricting clinical applicability.

## Research Motivation

Semi-supervised learning (SSL) offers a promising framework to leverage abundant unlabeled data alongside scarce labeled samples. Recent advances, such as FixMatch and consistency regularization, have demonstrated state-of-the-art improvements in image classification by effectively utilizing unlabeled data through:

- **Pseudo-labeling:** Assigning model-predicted labels to unlabeled samples, augmenting the training set.
- **Consistency regularization:** Enforcing model predictions to be invariant under input perturbations, improving robustness.

This project integrates these state-of-the-art SSL paradigms with domain-specific considerations, including patient-wise data splitting and class imbalance strategies, to develop a robust CNN-based histopathology classifier.

---

## Methodology

### Data Preparation & Preprocessing

- **Patient-wise split:** Ensures no patient data leakage between training, validation, and test sets, critical for realistic generalization evaluation. This approach mitigates overoptimistic performance arising from correlated images within the same patient.
- **Data augmentation:** Extensive augmentation using geometric (random rotations, flips, scaling) and photometric transformations (brightness, contrast adjustments) addresses intra-sample variance and staining heterogeneity. These augmentations serve as implicit regularizers and improve the model’s ability to generalize.
- **Class imbalance mitigation:**  
  - **SMOTE (Synthetic Minority Over-sampling Technique):** Generates synthetic minority class examples in feature space, counteracting skewed distributions.  
  - **Class-weighted loss functions:** Penalize misclassification of minority classes more heavily to balance gradient contributions during training.

### Model Architecture


A modified **VGG-inspired CNN** was employed, with the following key design choices:

| Component                | Description                              | Justification                                                                                  |
|--------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------|
| Convolutional blocks     | 3 blocks with increasing filters: 32 → 64 → 128 | Hierarchical feature extraction from low-level edges to complex patterns                     |
| Kernel size              | 3×3 convolutions with ‘same’ padding   | Preserves spatial resolution, captures fine cellular textures                                |
| Batch Normalization      | After each convolutional layer          | Stabilizes and accelerates training by normalizing layer inputs                              |
| Dropout                  | 0.3–0.4 in fully connected layers       | Reduces co-adaptation of neurons and mitigates overfitting                                  |
| L2 regularization        | λ=0.001 applied to convolutional and dense layers | Prevents large weight magnitudes, encouraging smoother decision boundaries                   |
| MaxPooling               | 2×2 pooling after each block             | Dimensionality reduction and translational invariance                                       |
| Output activation        | Sigmoid (binary), Softmax (multi-class) | Aligns with classification task objectives                                                 |
| Learning rate scheduler  | Cosine annealing                        | Smooth learning rate decay improves convergence and helps escape sharp local minima        |
| Early stopping           | Monitors validation loss and accuracy   | Prevents overfitting by halting training at optimal generalization point                    |

### Semi-Supervised Learning Implementation

- **Pseudo-labeling strategy:**  
  - Unlabeled images were passed through the model to obtain predicted class probabilities.  
  - Predictions exceeding a confidence threshold of 0.7 were accepted as pseudo-labels.  
  - This threshold balances precision of pseudo-labels against dataset expansion.  
- **Dataset expansion:** The original labeled dataset (~3,000 samples) was augmented by ~4,778 pseudo-labeled samples, resulting in an enlarged training set of ~11,944 images.  
- **Consistency regularization:** Augmented versions of the same input are encouraged to yield consistent predictions, enhancing model robustness to input perturbations.  
- **Fine-tuning:** The CNN was retrained on the combined labeled and pseudo-labeled datasets, allowing transfer of learned representations and improved generalization.

---

## Results & Analysis

| Metric              | Initial Model (Labeled Only) | Final Model (Labeled + Pseudo-Labels) |
|---------------------|-----------------------------|--------------------------------------|
| Accuracy            | ~78.5%                      | 84.0%                                |
| Macro Precision     | 75.0%                       | 82.1%                                |
| Macro Recall        | 74.8%                       | 83.5%                                |
| Macro F1 Score      | 74.9%                       | 82.7%                                |

- **Patient-wise validation splits** ensured robust evaluation without leakage.
- **Improved metrics** indicate enhanced model generalization and ability to detect minority cancer classes.
- The **confidence distribution histogram** of pseudo-labels confirmed that a 0.7 threshold filters out low-confidence (noisy) labels, increasing dataset reliability.
- **Class distribution comparison before and after pseudo-labeling** revealed improved representation of minority classes, though some imbalance persists (e.g., missing Type 1 cells in pseudo-labels).

### Pseudo-labeling Unlabeled Data

- **Total unlabeled samples:** 10,384  
- **High-confidence pseudo-labels:** 4,778 (~46%)  

**Class distribution of pseudo-labels:**  
- Type 0: 92.72% (4430)  
- Type 1: 0% (0)  
- Type 2: 6.11% (292)  
- Type 3: 1.17% (56)  

**Issue:** Severe imbalance; Type 1 completely missing, risking bias and poor minority class learning.

### Combined Dataset

- Original labeled data: 7,166  
- Pseudo-labeled data: 4,778  
- **Total for fine-tuning:** 11,944  

Combining increased data size to improve generalisation and reduce overfitting.

### Training Summary

- Validation accuracy peaked at epoch 7.  
- Training accuracy continued rising, indicating overfitting.  
- Loss and validation metrics stopped improving, showing limited generalisation gains.

---

## Limitations & Recommendations

- Pseudo-labeling improved accuracy but introduced class imbalance risks due to missing classes in pseudo-labeled data.
- Overfitting observed as training accuracy outpaced validation accuracy after certain epochs.
- High-confidence threshold may exclude difficult but informative samples, limiting minority class learning.

### Recommendations for Improvement

1. **Adaptive Class-Specific Confidence Thresholds:**  
   Dynamically adjust thresholds per class to ensure balanced pseudo-labeling.

2. **Advanced Loss Functions:**  
   Implement focal loss or class-balanced loss to emphasize minority classes and difficult samples.

3. **Uncertainty Estimation:**  
   Use Bayesian neural networks or Monte Carlo dropout to measure prediction uncertainty for improved pseudo-label filtering.

4. **Mean Teacher and FixMatch++ Architectures:**  
   Employ teacher-student models for stable pseudo-label generation and improved consistency.

5. **Self-Supervised Pretraining:**  
   Leverage contrastive learning frameworks (SimCLR, MoCo) on unlabeled data to learn robust representations.

6. **Explainability:**  
   Integrate Grad-CAM or Integrated Gradients to provide model interpretability for clinical validation.

7. **Multi-Task Learning:**  
   Extend model to simultaneously predict cancer presence and cell subtype for richer feature learning.

---

## Appendix: Patient-wise Data Independence

Patient-wise splitting is critical to prevent data leakage. Since multiple images per patient are similar, random splits may cause overoptimistic test results. Ensuring patient-level data isolation simulates real-world deployment and improves model robustness.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{kritiyadav2025cancer,
  title={Semi-Supervised CNNs for Histopathological Cancer Cell Classification},
  author={Kriti Yadav},
  year={2025},
  url={https://github.com/Kriti-Data-Business/histopathology-cancer-cnn-semisupervised}
}
````

---

## Acknowledgements

Special thanks to the medical imaging and open-source machine learning communities, and the researchers whose foundational work inspired this project:

* Google Research:

  * [Semi-Supervised Learning with Consistency Regularization (2023)](https://research.google/pubs/pub50610/)
  * [FixMatch: Simplifying Semi-Supervised Learning (2020)](https://arxiv.org/abs/2001.07685)
* Academic Papers:

  * [Pseudo-Label: Efficient Semi-Supervised Learning (Lee, 2013)](https://www.cs.utoronto.ca/~tingwuwang/teaching/2017/CSC2541/lectures/PseudoLabel.pdf)
  * [Self-Training for Medical Imaging (Zhu et al., 2021)](https://arxiv.org/abs/2006.03950)
  * [Semi-Supervised and Few-Shot Learning for Medical Imaging (Yang et al., 2022)](https://www.nature.com/articles/s41746-022-00683-3)

---

## Dataset links 
1. https://ieeexplore.ieee.org/document/7312934

2. https://github.com/maduc7/Histopathology-Datasets

Feel free to reach out if you want to contribute or discuss improvements!
