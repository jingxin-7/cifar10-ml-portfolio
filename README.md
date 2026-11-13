# CIFAR-10 with classical machine learning

**68.24% mean official course validation accuracy** across three independent training seeds. This project classifies 32 × 32 RGB images into ten classes using learned patch features and scikit-learn classifiers. The goal was to improve accuracy under image corruptions without using pretrained models or external training data.

This portfolio documents the completed Phase 7R system. It includes a small, synthetic-data demonstration of the feature encoding and fusion. The fitted models and complete submission stay in the private archive.

## Feature extraction

For the fine branches, I applied deterministic pixel repair and smoothing, extracted 6 × 6 RGB patches at stride 1, normalized each patch, and applied regularized ZCA whitening fitted on training patches. A 512-centroid MiniBatchKMeans dictionary learned local visual patterns. Both fine branches used the same dictionary and appended 2,148 HOG and color features.

| Representation | How it works | Features |
| --- | --- | ---: |
| Fine triangle | A centroid is active when its distance from a patch is below that patch's mean distance to all centroids. Mean and max responses are pooled globally and in four quadrants. | 7,268 |
| Fine polarity | Whitened patches are projected onto the same centroids. Positive and negative responses above a threshold of 0.5 are pooled separately with the same mean and max statistics. | 12,388 |
| Coarse triangle | 2 × 2 area averaging reduces the image to 16 × 16. A separate training-fitted dictionary produces stride-1 triangle features; the pooled values receive a square-root transform. | 5,120 |

The two resolutions serve different roles. The fine branches retain local gradients and patch detail; the coarse branch tests whether pooled shape and color structure offer complementary evidence.

## Classifiers and probability ensemble

The fine triangle descriptor feeds a 256-unit MLP, a 128-component PCA projection followed by histogram gradient boosting, and an averaged logistic SGD classifier. Separate averaged logistic SGD classifiers use fine polarity and coarse triangle features. Scaling and other fitted transforms use training data only.

The final decision averages aligned ten-class probability vectors with fixed weights:

```text
inner   = 0.50 × MLP   + 0.50 × PCA-tree
phase_h = 0.50 × inner + 0.50 × fine triangle SGD
phase_p = 0.75 × phase_h + 0.25 × fine polarity SGD
final   = 0.75 × phase_p + 0.25 × coarse triangle SGD
```

Each estimator predicts from one image view at inference. Its class is the argmax of the fused probabilities. The three seed models are independent runs; the reported score averages their accuracies.

## Methodology and results

Development comparisons used disjoint fitting and evaluation images. The final coarse-fusion pilot used 35,000 fitting images, 5,000 clean development images, and 1,500 linked source images across 27 corruption conditions. Balanced development accuracy averaged the clean accuracy and the mean corruption accuracy equally.

| Development metric | Phase P | Phase R |
| --- | ---: | ---: |
| Clean accuracy | 79.50% | 79.48% |
| Mean corrupted accuracy | 69.635% | 70.452% |
| Balanced accuracy | 74.567% | 74.966% |

The balanced gain was **0.399 percentage points** (paired 95% bootstrap interval: **+0.072 to +0.737**). It missed the original 1-point promotion threshold. I later authorized a single evaluation of the frozen configuration; this was an explicit exception to the development criterion.

Final training used the 50,000 authorized training images, with the original images and two fixed augmented views. The official course validation set contained 5,000 images.

| Training seed | Phase P validation | Phase R validation |
| --- | ---: | ---: |
| 42 | 67.20% | 68.14% |
| 2026 | 67.38% | 68.46% |
| 3102 | 67.58% | 68.12% |
| **Mean** | **67.39%** | **68.24%** |

Phase R improved the mean by **0.85 percentage points**. A separate fresh-kernel run matched all 15,000 recorded validation predictions across the three seeds. These are historical results from the private submission. The portfolio demo does not reproduce or remeasure them.

## Limitations and what I learned

The 68.24% figure is specific to the course validation split. It is not the standard CIFAR-10 test score, a hidden-test score, or a class ranking. Repeated use of the development suite makes the bootstrap interval conditional on the compared predictors; it does not account for all earlier selection decisions. Clean and corrupted cross-entropy both worsened in the coarse-fusion pilot, even though mean corrupted accuracy improved. An untouched evaluation split and calibration checks would make a follow-up comparison stronger.

## Run the portfolio demo

With Python 3 and NumPy installed:

```bash
python examples/synthetic_demo.py
```

The demo uses generated arrays to check triangle encoding, polarity splitting, multiscale pooling, and probability fusion. It downloads nothing, fits nothing, and writes no files. See [Methodology](docs/methodology.md) for implementation details, evaluation boundaries, and the historical software environment.

## AI-assisted development

I used AI assistance to develop and debug the implementation, structure controlled comparisons, automate reproducibility checks, and draft documentation. I chose which experiments to pursue and used matching splits, fixed configurations, recorded metrics, and separate verification runs to assess changes. The course allowed AI assistance and made students responsible for understanding and verifying their submissions. This portfolio records that assistance without including private conversations or course materials.

## References and reuse

- [CIFAR-10 dataset and technical-report citation](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
- [Coates, Ng, and Lee, 2011](https://proceedings.mlr.press/v15/coates11a.html), for the patch dictionary and triangle-encoding approach.
- [Dalal and Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), for histograms of oriented gradients.

The references credit method and dataset sources. This portfolio includes a newly written demonstration and does not bundle datasets, third-party library code, or the graded submission. No open-source license has been selected.
