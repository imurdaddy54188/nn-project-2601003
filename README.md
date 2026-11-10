# Construction Vehicle Classification with a CNN

BAI13143 Neural Network Basics – Individual Project

A convolutional neural network (designed from scratch in PyTorch) that classifies
cropped images of construction vehicles into 6 classes:
`bulldozer`, `concrete_mixer`, `crane`, `excavator`, `loader`, `truck`.

Training and testing run in two separate Docker containers. The test container has
no PyTorch at all: it runs the exported ONNX model with onnxruntime.

## Dataset

- **ACID_MOCS 1** by SODA1, Roboflow Universe – https://universe.roboflow.com/soda1/acid_mocs-1/dataset/1
- Licence: **CC BY 4.0**
- Exported in YOLOv8 format (object detection). `src/prepare_data.py` crops every
  bounding box into a classification dataset. Roboflow's own train/valid/test split is
  kept, so crops from one photo never appear in two splits.

| split | per class | total |
|---|---|---|
| train | 800 | 4,800 |
| valid | 150 | 900 |
| test  | 126–150 | 858 |

## Project structure

```
nn-project-<studentID>/
├── data/
│   ├── raw/              # Roboflow export (not in git)
│   ├── crops/            # output of prepare_data.py (not in git)
│   └── new/              # your own photos for predict.py
├── models/               # model.onnx, model.pt, config.json (shared via volume)
├── outputs/              # charts, curves, experiment log, test metrics
├── src/
│   ├── prepare_data.py   # YOLO boxes -> class folders
│   ├── explore_data.py   # EDA charts + cleaning checks
│   ├── preprocess.py     # preprocessing shared by train and test
│   ├── model.py          # VehicleCNN architecture
│   ├── train.py          # training, experiments, ONNX export
│   ├── test.py           # reproducibility check + test-set evaluation
│   └── predict.py        # predict new images
├── Dockerfile.train
├── Dockerfile.test
├── requirements-train.txt
└── requirements-test.txt
```

## 1. Prepare the data (local, once)

```powershell
pip install pillow pyyaml pandas matplotlib
python src/prepare_data.py
python src/explore_data.py
```

## 2. Train with Docker

```powershell
docker build -f Dockerfile.train -t nn-train .

# final model (default command)
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" -v "${PWD}/outputs:/app/outputs" nn-train

# experiments: pass the arguments after the image name
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" -v "${PWD}/outputs:/app/outputs" nn-train --name exp0_baseline --note "baseline"
```

Training writes `models/model.onnx`, `models/model.pt` and `models/config.json`
(class names, image size and the training mean/std used to standardise images).

## 3. Test with Docker (on a different machine)

Copy the project folder **including `data/crops` and `models/`** to the second machine, then:

```powershell
docker build -f Dockerfile.test -t nn-test .
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" -v "${PWD}/outputs:/app/outputs" nn-test
```

The first lines compare the validation accuracy measured in this container with the value
recorded by the training container (`RESULT: MATCH`). Then the test-set metrics are printed and
the confusion matrix and wrong-prediction grid are saved to `outputs/test/`.

## 4. Predict a new image

Put photos in `data/new/`, then:

```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" nn-test python src/predict.py data/new
```

## Reproducibility

- All library versions are pinned in the two requirements files.
- Seeds are fixed (`random`, `numpy`, `torch`, `PYTHONHASHSEED`) and
  `torch.use_deterministic_algorithms(True)` is on.
- Files are always read in sorted order.
- The test container re-checks validation accuracy against the value saved at training time.

## Use of generative AI

Claude (Anthropic) was used to help write and debug the scripts and Docker file.
All code was run, checked and modified by the author.
https://drive.google.com/file/d/1G_3agke0eKaidZo5U4p9HcCzkq4OPlNr/view?usp=drive_link
