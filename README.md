# SatSeg (WIP)
A python package to assist in multispectral satellite image segmentation.

## Installation
```
pip install satseg
```

## Usage

### Training
```python
from satseg.dataset import create_datasets
from satseg.model import save_model, train_model

# A list of paths to the satellite image tif files (.TIF)
tif_paths = ["01.tif". "02.tif"]
# A list of paths to the shapefiles (.SHP) of the masks
mask_paths = ["mask.shp"]

# Create the train and validation datasets
# The intermediate image files will be saved at ./temp/labeled/
train_set, val_set = create_datasets(tif_paths, mask_paths, "temp/labeled")

# Train the model
model, metrics = train_model(train_set, val_set, "unet")

# Save the model at a specified path
model_path = "model.pt"
save_model(model, model_path)
```


### Inference
```python
from satseg.dataset import create_inference_dataset
from satseg.model import load_model, run_inference


tif_paths = ["03.tif", "04.tif"]
# The intermediate image files will be saved at ./temp/unlabeled/
dataset = create_inference_dataset(tif_paths, "temp/unlabeled/", 256)

# Load a previously saved model
model_path = "model.pt"
model = load_model(model_path)

# Run inference on the dataset using the loaded model
# Results will be saved ./temp/infer/
run_inference(dataset, model, "temp/infer/")

```
