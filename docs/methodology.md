# Methodology notes

## From image to descriptors

For each training run, normalization subtracts the patch mean and divides by the square root of the patch variance plus 10. The resulting 108-dimensional patch vector is centered using the training patch mean, then multiplied by a regularized ZCA whitening matrix. The coarse dictionary is learned separately from the fine dictionary; both contain 512 centroids.

For whitened patch $u$ and centroid $c_k$, triangle activation is $\max(0, \overline{d} - d_k)$, where $d_k = \lVert u - c_k\rVert_2$ and $\overline{d}$ is the mean over centroids for that patch. The polarity branch separately retains $\max(0, u\cdot c_k - 0.5)$ and $\max(0, -u\cdot c_k - 0.5)$. Global and quadrant means and maxima give ten statistics per response channel. Fine features append HOG and RGB summaries; the coarse branch uses only its pooled response features. The synthetic example illustrates these operations with four generated centroids and no fitted preprocessing.

## Training and evaluation discipline

- Preprocessing, patch dictionaries, standardization, and PCA were fitted using the training data. No pretrained model or external training dataset was used.
- The final neural classifier used one 256-unit hidden layer with ReLU and Adam. The tree head used 128-component PCA and histogram gradient boosting. The SGD classifiers used logistic loss and averaged weights beginning after the first epoch.
- Development comparisons retained the same splits and corruption conditions. Paired bootstrap resampling kept an image's linked corruption observations together; the reported interval captures uncertainty on that development suite for the two compared predictors.
- The final configuration was frozen before official validation scoring. The historical submission's separate full reproduction run verified seed-by-seed prediction parity. This portfolio is a method illustration and results record, not an end-to-end reproduction package.

## Historical environment

| Component | Version used for verification |
| --- | --- |
| Python | 3.14.6 |
| NumPy | 2.4.6 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| joblib | 1.5.3 |

The synthetic example only requires NumPy. No serialization format or fitted artifact is required to run it.
