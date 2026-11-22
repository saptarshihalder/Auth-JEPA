# Auth-JEPA

Detecting AI-generated images by predicting camera physics, trained on real photographs only.

| Folder | What it shows |
|---|---|
| `01_patches` | 512 x 512 window, never resized, cut into 16 patches of 128 x 128 |
| `02_denoiser` | wavelet-domain Wiener filter: smoothed patch and removed noise |
| `03_inputs` | the 15 inputs: 13 picture measurements and 2 JPEG values |
| `04_noise_measurements` | the 74 noise measurements (45 original + 29 added) |
| `05_predictor` | the predictor network |
| `06_training_and_score` | Gaussian loss, training, patch score and image score |
| `07_test_conditions` | raw and JPEG-90 test images |
| `08_auroc` | AUROC per generator, then averaged |
| `09_two_models_threshold` | two models, calibration and the fixed cut-off |
| `dataset` | file lists and example images for every dataset and generator |
| `output` | result tables and output images, by slide |

## Loss function and score

Auth-JEPA is trained with the **Gaussian negative log-likelihood (NLL)**. For every patch $t$ of an image, the predictor gives a mean $\mu_{t,d}$ and a variance $\sigma^2_{t,d}$ for each of the 74 noise measurements $d$, and the loss compares them with the measured value $\phi_{t,d}$:

$$
\mathcal{L}_t = \frac{1}{2} \sum_{d=1}^{74} \left[ \log \sigma^2_{t,d} + \frac{(\phi_{t,d} - \mu_{t,d})^2}{\sigma^2_{t,d}} \right]
$$

The first term stops the network from predicting a large spread everywhere; the second penalises a wrong prediction, scaled by the expected spread. Training minimises the mean of $\mathcal{L}_t$ over all patches of the real training photos. No AI-generated image is used.

The **image score** is the same quantity, averaged over the 16 patches of a $512 \times 512$ window:

$$
S(x) = \frac{1}{16} \sum_{t=1}^{16} \mathcal{L}_t
$$

A low score means the noise looks like a real camera's; a high score flags the image as AI-generated. Both are implemented by `tile_nll` in `06_training_and_score/loss_and_score.py`.

| Symbol | Meaning |
|---|---|
| $x$ | input image |
| $t$ | patch, 1 to 16 |
| $d$ | noise measurement, 1 to 74 |
| $\phi_{t,d}$ | measured value |
| $\mu_{t,d}$ | predicted mean |
| $\sigma^2_{t,d}$ | predicted variance |
| $\mathcal{L}_t$ | patch score (loss) |
| $S(x)$ | image score |

