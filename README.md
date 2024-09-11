# Siapy command line tool

This repository provides a command line interface (CLI) for the [siapy](https://github.com/siapy/siapy-lib) library, designed to streamline the segmentation of spectral images for further analysis.

With this CLI, you can:

- Display images from two cameras.
- Co-register cameras and compute the transformation from one camera's space to another.
- Select regions in images for training machine learning (ML) models.
- Perform image segmentation using a pre-trained ML model.
- Convert radiance images to reflectance by utilizing a reference panel.
- Display spectral signatures for in-depth analysis.

## 🏃‍♀️ Installation

1. Clone the Repository and Install Dependencies

Start by cloning the repository and running the installation script:

``` zsh
git clone https://github.com/siapy/siapy-cli.git
cd siapy-cli
./scripts/install-dev.sh
```

2. Configure Environment Variables

Create `.env` file in the root of the project directory (inside `siapy-cli` folder) and define the necessary environment variables:

``` env
# -------------------------------------------------------------
# .env file located in the root of the project repository

# Set the project name, e.g.:
PROJECT_NAME=example_project

# Specify the directory where spectral images are stored, e.g.:
IMAGES_DIR=/path/to/your/spectral_images
# -------------------------------------------------------------
```

> :exclamation: **Note for WSL (Windows Subsystem for Linux) users:**
>
> - Make sure that images are also located within the WSL file system, e.g. in `/home/$USER/data/` directory
> - To easily access your WSL file system from Windows, open `explorer.exe` and type `\\wsl$\Ubuntu` into the address bar to navigate your WSL files.
>

3. Verify the Installation

Run one of the following commands to verify the installation and check if everything is working:

``` zsh
siapy-cli --version
# or
pdm run main.py --version
# or
python main.py --version
# ...
```

## 🚀 Usage

```
$ siapy-cli --help

Usage: siapy-cli [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion    Install completion for the current shell.
  --show-completion       Show completion for the current shell, to copy it or customize the installation.
  --help                  Show this message and exit.

Commands:
  calculate-transformation
  check-images
  convert-to-reflectance
  create-signatures
  display-image
  display-settings
  segment-images
  select-areas
  train-model
```

## 📖 Cookbook

This guide provides a step-by-step workflow to segment relevant areas of spectral images.

**Image Naming Convention**

Before you begin, ensure that the images are correctly named according to the following convention:

- Image file: `L1_L2_L3__*.img`
- Header file: `L1_L2_L3__*.hdr` (corresponding to the image file)

Where:

- `L1`, `L2`, `L3`, etc., represent labels for objects in the spectral image.
- The number of labels (`L`) can vary depending on the number of objects in the image.
- Labels are separated by an underscore (`_`)
- Double underscore (`__`) separates the label section from the rest of the filename.

### Workflow

1. Check images

Run the following command to check the images:

``` zsh
siapy-cli check-images
```

- The number of images and unique labels should be the same for both cameras.
- The duplicated labels space should be empty.

2. Calculate transformation between cameras

Calculate the transformation between the two cameras using the label L:

``` zsh
siapy-cli calculate-transformation L
```

- L is the label on one image where the corresponding points will be selected first on camera one and then on camera two.
- Select at least 6 points, but preferably more than 9.
- Try to select the same positions on matching images as accurately as possible.

3. Select areas for ML model training

Run the following commands to select approximately balanced areas for each category (object or background):

``` zsh
# For label e.g. object
siapy-cli select-areas L object
# For label e.g. background
siapy-cli select-areas L background
```

4. Train model based on selected areas

``` zsh
siapy-cli train-model
```

5. Segment images

If all the steps were executed successfully, you can proceed to segment the images:

``` zsh
# Start from the beginning
siapy-cli segment-images
# Start from label L
siapy-cli segment-images L
```

- First, select the reference panel.
- Then, select all the objects and press enter.
- The segmentation masks are drawn; press save to save the segmented image, repeat to repeat the process, and skip to proceed to the next one without saving.

6. Convert to reflectance

Convert images based on reference panel reflectance values:

``` zsh
siapy-cli convert-to-reflectance VALUE
# e.g. for reflectance value of 0.2
siapy-cli convert-to-reflectance 0.2
```

7. Convert to spectral signatures

Convert the segmented images to a tabular format for further analysis:

``` zsh
siapy-cli create-signatures
```

- This step will create one row for each object. Therefore, one object in the image will be described by one spectral signature.

### Output

Upon execution, images and a Parquet file will be created. All artifacts are saved in `siapy-cli/artifacts` directory.

The columns of the Parquet file represent the following:

- **filename**: 📄 The name of the image file from which the pixel originates.
- **label**: 🏷️ The label assigned to the image.
- **image_idx**: 🔢 The index of the image in the dataset.
- **object_idx**: 🔢 The index of the object within the image.
- **signature**: 📈 The spectral values associated with the object.

Example row:

```json
{
  "filename": "L1_L2_L3",
  "label": "L1",
  "image_idx": "0",
  "object_idx": "1",
  "signature": [...]
  }
```
