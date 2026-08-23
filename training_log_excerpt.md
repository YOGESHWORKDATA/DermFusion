```
DEVICE: cuda
GPU: Tesla T4 | 15.6 GB
HAM10000: 10015 images, 7 classes
Class distribution:
label
0    1113
1    6705
2     514
3     327
4    1099
5     115
6     142
Train: 8012 | Valid: 2003

Building DermFusion...
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth

```

```
100%|██████████| 97.8M/97.8M [00:00<00:00, 215MB/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

```

Error displaying widget: model not found

```
✓ Parameters: 137.0M
✓ Class weights: [0.277 0.046 0.6   0.943 0.281 2.682 2.172]
✓ Backbone frozen for warmup
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 1/45
  ACC    : 0.7384    AUC    : 0.9275
  F1     : 0.5126    DICE   : 0.8689
  UNC    : 0.001870
  MEL Sensitivity : 0.2466
  Selective ACC   : 0.7667 (τ=0.002129, coverage=68.05%)
  v1=0.0034  v2=-0.0271
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 2/45
  ACC    : 0.7873    AUC    : 0.9399
  F1     : 0.6213    DICE   : 0.8964
  UNC    : 0.001964
  MEL Sensitivity : 0.3722
  Selective ACC   : 0.8503 (τ=0.002236, coverage=68.05%)
  v1=-0.0287  v2=-0.0846
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 3/45
  ACC    : 0.7604    AUC    : 0.9422
  F1     : 0.5736    DICE   : 0.9018
  UNC    : 0.002016
  MEL Sensitivity : 0.4529
  Selective ACC   : 0.7880 (τ=0.002143, coverage=68.05%)
  v1=-0.0784  v2=-0.1678
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 4/45
  ACC    : 0.8023    AUC    : 0.9519
  F1     : 0.6546    DICE   : 0.9110
  UNC    : 0.001730
  MEL Sensitivity : 0.4798
  Selective ACC   : 0.8503 (τ=0.002004, coverage=68.05%)
  v1=-0.1290  v2=-0.2438
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 5/45
  ACC    : 0.7963    AUC    : 0.9550
  F1     : 0.6712    DICE   : 0.9191
  UNC    : 0.001980
  MEL Sensitivity : 0.4350
  Selective ACC   : 0.8430 (τ=0.002250, coverage=68.05%)
  v1=-0.1713  v2=-0.3177
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 6/45
  ACC    : 0.7803    AUC    : 0.9550
  F1     : 0.6494    DICE   : 0.9219
  UNC    : 0.002272
  MEL Sensitivity : 0.5830
  Selective ACC   : 0.8305 (τ=0.002300, coverage=68.05%)
  v1=-0.2159  v2=-0.3894
============================================================
  ★ BEST MODEL SAVED

============================================================
  BACKBONE UNFROZEN at epoch 7
============================================================
  [morph_strength = 0.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 7/45
  ACC    : 0.8298    AUC    : 0.9651
  F1     : 0.7359    DICE   : 0.9222
  UNC    : 0.001599
  MEL Sensitivity : 0.5785
  Selective ACC   : 0.8863 (τ=0.001735, coverage=68.05%)
  v1=-0.2821  v2=-0.4572
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.17]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 8/45
  ACC    : 0.8228    AUC    : 0.9708
  F1     : 0.7507    DICE   : 0.9269
  UNC    : 0.001299
  MEL Sensitivity : 0.7354
  Selective ACC   : 0.9083 (τ=0.001494, coverage=68.05%)
  v1=-0.3620  v2=-0.5206
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.33]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 9/45
  ACC    : 0.8562    AUC    : 0.9782
  F1     : 0.7835    DICE   : 0.9284
  UNC    : 0.001198
  MEL Sensitivity : 0.6816
  Selective ACC   : 0.9230 (τ=0.001331, coverage=68.05%)
  v1=-0.4339  v2=-0.5791
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.50]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 10/45
  ACC    : 0.8198    AUC    : 0.9760
  F1     : 0.7625    DICE   : 0.9305
  UNC    : 0.001116
  MEL Sensitivity : 0.7578
  Selective ACC   : 0.8767 (τ=0.001226, coverage=68.05%)
  v1=-0.4985  v2=-0.6357
============================================================
  [morph_strength = 0.67]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 11/45
  ACC    : 0.8727    AUC    : 0.9829
  F1     : 0.8129    DICE   : 0.9282
  UNC    : 0.001113
  MEL Sensitivity : 0.7668
  Selective ACC   : 0.9391 (τ=0.001214, coverage=68.05%)
  v1=-0.5622  v2=-0.6903
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 0.83]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 12/45
  ACC    : 0.8832    AUC    : 0.9829
  F1     : 0.8411    DICE   : 0.9283
  UNC    : 0.000955
  MEL Sensitivity : 0.7399
  Selective ACC   : 0.9457 (τ=0.001011, coverage=68.05%)
  v1=-0.6190  v2=-0.7364
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 13/45
  ACC    : 0.8822    AUC    : 0.9828
  F1     : 0.8383    DICE   : 0.9269
  UNC    : 0.000868
  MEL Sensitivity : 0.7309
  Selective ACC   : 0.9618 (τ=0.000974, coverage=68.05%)
  v1=-0.6750  v2=-0.7779
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 14/45
  ACC    : 0.8612    AUC    : 0.9815
  F1     : 0.8172    DICE   : 0.9350
  UNC    : 0.001011
  MEL Sensitivity : 0.7489
  Selective ACC   : 0.9237 (τ=0.001068, coverage=68.05%)
  v1=-0.7218  v2=-0.8131
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 15/45
  ACC    : 0.8952    AUC    : 0.9832
  F1     : 0.8423    DICE   : 0.9307
  UNC    : 0.000846
  MEL Sensitivity : 0.7399
  Selective ACC   : 0.9640 (τ=0.000902, coverage=68.05%)
  v1=-0.7665  v2=-0.8496
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 16/45
  ACC    : 0.9051    AUC    : 0.9851
  F1     : 0.8537    DICE   : 0.9323
  UNC    : 0.000716
  MEL Sensitivity : 0.7309
  Selective ACC   : 0.9714 (τ=0.000767, coverage=68.05%)
  v1=-0.8121  v2=-0.8812
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 17/45
  ACC    : 0.8917    AUC    : 0.9827
  F1     : 0.8555    DICE   : 0.9330
  UNC    : 0.000890
  MEL Sensitivity : 0.8206
  Selective ACC   : 0.9494 (τ=0.000799, coverage=68.05%)
  v1=-0.8531  v2=-0.9113
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 18/45
  ACC    : 0.8962    AUC    : 0.9845
  F1     : 0.8529    DICE   : 0.9355
  UNC    : 0.000778
  MEL Sensitivity : 0.8430
  Selective ACC   : 0.9589 (τ=0.000784, coverage=68.05%)
  v1=-0.8930  v2=-0.9365
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 19/45
  ACC    : 0.9111    AUC    : 0.9877
  F1     : 0.8643    DICE   : 0.9357
  UNC    : 0.000627
  MEL Sensitivity : 0.6816
  Selective ACC   : 0.9758 (τ=0.000636, coverage=68.05%)
  v1=-0.9267  v2=-0.9581
============================================================
  ★ BEST MODEL SAVED
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 20/45
  ACC    : 0.8957    AUC    : 0.9877
  F1     : 0.8637    DICE   : 0.9349
  UNC    : 0.000678
  MEL Sensitivity : 0.8700
  Selective ACC   : 0.9677 (τ=0.000667, coverage=68.05%)
  v1=-0.9575  v2=-0.9786
============================================================
  [morph_strength = 1.00]

```

Error displaying widget: model not found

```
============================================================
  EPOCH  : 21/45
  ACC    : 0.9136    AUC    : 0.9872
  F1     : 0.8778    DICE   : 0.9306
  UNC    : 0.000592
  MEL Sensitivity : 0.7713
  Selective ACC   : 0.9773 (τ=0.000535, coverage=68.05%)
  v1=-0.9848  v2=-0.9965
============================================================
  [morph_strength = 1.00]
```