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
