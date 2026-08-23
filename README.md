# Data

The dataset is **not included** in this repository.

The paper uses:

- HAM10000: 10,015 dermoscopic images, seven diagnostic classes.
- ISIC 2018 Task 1 lesion-boundary annotations for segmentation masks.
- External evaluation: PH2, ISIC 2019, and PAD-UFES-20.

The training implementation expects the HAM10000 segmentation/classification package to contain:

```text
ham1000-segmentation-and-classification/
├── GroundTruth.csv
├── images/
└── masks/
```

`GroundTruth.csv` must contain the seven one-hot class columns used by the implementation and an `image` identifier matching the image/mask filenames.

Do not commit the dataset, patient metadata, or trained weights to this repository unless their redistribution terms explicitly permit it.
