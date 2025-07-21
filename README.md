# Siapy CLI

This repository provides a command line interface (CLI) for the [siapy](https://github.com/siapy/siapy-lib) library.

> :exclamation: **Note**: This repository was developed for academic purposes and serves as a segmentation tool for hyperspectral data preparation, machine learning model training, and hypothesis testing. It demonstrates how the siapy library can be used for these and similar purposes. Please note that while the siapy library has progressed since this implementation, this CLI uses an earlier version for compatibility with the academic research it was designed to support.

With this CLI, you can:

- Display images from two cameras.
- Co-register cameras and compute the transformation from one camera's space to another.
- Select regions in images for training machine learning (ML) models.
- Perform image segmentation using a pre-trained ML model.
- Convert radiance images to reflectance by utilizing a reference panel.
- Convert segmented areas into spectral signatures.
- Display spectral signatures.
- Build a machine learning model using these spectral signatures.
- Evaluate the model, generate metrics, and display results.

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
siapy-cli info --version
# or
pdm run ./source/main.py info --version
# or
python ./source/main.py info --version
# ...
```

## 🚀 Usage

``` zsh
$ siapy-cli --help

Usage: siapy-cli [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion    Install completion for the current shell.
  --show-completion       Show completion for the current shell, to copy it or customize the installation.
  --help                  Show this message and exit.

Commands:
  info            General information about the project or the environment.
  misc            Miscellaneous commands, e.g., check images, statistics, etc.
  segment         Segmentation commands, e.g., select areas, train model, etc.
  analysis        Analysis commands, e.g., train model, generate metrics, etc.
```

During execution, all artifacts (including required data, transformed images, extracted signatures, models, etc.) are saved in the `siapy-cli/artifacts` directory.

## 📖 Cookbook

This guide provides an opinionated step-by-step workflow for using the `siapy-cli` tool.
To follow along, download the example data from [Zenodo](https://zenodo.org/records/14534998).

**Image Naming Convention**

Before you begin, ensure that the images are correctly named according to the following convention:

- Image file: `L1_L2_L3__*.img`
- Header file: `L1_L2_L3__*.hdr` (corresponding to the image file)

Where:

- `L1`, `L2`, `L3`, etc., represent labels for objects in the spectral image.
- The number of labels (`L`) can vary depending on the number of objects in the image.
- Labels are separated by an underscore (`_`)
- Double underscore (`__`) separates the label section from the rest of the filename.

---

### Workflow - Segmentation of images

1. Check images

Run the following command to check the images:

``` zsh
siapy-cli misc check-images
```

- The number of images and unique labels should be the same for both cameras.
- The duplicated labels space should be empty.

2. Calculate transformation between cameras

Calculate the transformation between the two cameras using the label L:

``` zsh
siapy-cli segment calculate-transformation L
```

- L is the label on one image where the corresponding points will be selected first on camera one and then on camera two.
- Select at least 6 points, but preferably more than 9.
- Try to select the same positions on matching images as accurately as possible.

3. Select areas for ML model training

Run the following commands to select approximately balanced areas for each category (object or background):

``` zsh
# For label e.g. object
siapy-cli segment select-areas L object
# For label e.g. background
siapy-cli segment select-areas L background
```

4. Train model based on selected areas

``` zsh
siapy-cli segment train-model
```

5. Segment images

If all the steps were executed successfully, you can proceed to segment the images:

``` zsh
# Start from the beginning
siapy-cli segment segment-images
# Start from label L
siapy-cli segment segment-images --label L
```

- First, select the reference panel.
- Then, select all the objects and press enter.
- The segmentation masks are drawn; press save to save the segmented image, repeat to repeat the process, and skip to proceed to the next one without saving.

6. Convert to reflectance

Convert images based on reference panel reflectance values:

``` zsh
siapy-cli segment convert-to-reflectance VALUE
# e.g. for reflectance value of 0.2
siapy-cli segment convert-to-reflectance 0.2
```

7. Convert to spectral signatures

Convert the segmented images to a tabular format for further analysis:

``` zsh
siapy-cli segment create-signatures
```

- This step will create one row for each object. Therefore, one object in the image will be described by one spectral signature.

**Output**

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

---

### Workflow - Model training and classification

**Configuration**

For comprehensive CLI utilization, users can define the model, dataset, and hyperparameters to optimize the model.

Extend functionality by implementing corresponding classes in `siapy-cli/extensions/` directory. Example implementations, prefixed with `ex_` (stands for example), are provided. Start from these examples and customize according to your requirements. Ensure compatibility with the CLI by inheriting from the same base classes. Newly added files are configured not to be tracked by git.

Once implemented, use these classes by specifying `--data-loader`, `--model`, and `--parameters` arguments in CLI commands.

> :exclamation: **Note**: Use method names, not file names, when calling newly implemented classes. For example, to use `DataLoaderExample`, use `--data-loader DataLoaderExample` (defined in ex_data_loader.py).

**Dummy example**

1. Test data load

Verify proper data loading:

``` zsh
siapy-cli analysis test-load-data --data-loader DataLoaderExample
```

- This command returns an array of signatures and their corresponding labels.
- Use `--data-loader` to specify the data loader class.
- Ensure the number of signatures matches the number of labels.

2. Train and optimize model

Train a model and optimize its hyperparameters:

``` zsh
siapy-cli analysis train-model --model SavgolPLSSVC \
                               --data-loader DataLoaderExample \
                               --parameters ParamsSavgol \
                               --parameters ParamsPLS \
                               --parameters ParamsSVC \
                               --do-optimize
```

- The `--do-optimize` flag enables hyperparameter optimization during training.
- You can customize and add new parameters to tune using the `--parameters` argument.
- Use the `--model` argument to select a different model.
- Ensure that the model and parameter configurations are compatible.

Once this step is complete, proceed to the following steps (3–6) using the same flags and arguments to maintain consistency across the workflow.

3. Generate metrics

Create performance metrics for the trained model:

``` zsh
siapy-cli analysis generate-metrics --model SavgolPLSSVC \
                                    --data-loader DataLoaderExample \
                                    --do-optimize
```

4. Generate plots

Produce visualizations for the analysis:

``` zsh
siapy-cli analysis generate-plots --model SavgolPLSSVC \
                                  --data-loader DataLoaderExample \
                                  --do-optimize
```

5. Calculate relevances

Analyze feature relevances:

``` zsh
siapy-cli analysis calculate-relevances --model SavgolPLSSVC \
                                        --data-loader DataLoaderExample \
                                        --do-optimize
```

6. Display metrics

Display the calculated metrics in the terminal:

``` zsh
siapy-cli analysis display-metrics --model SavgolPLSSVC --data-loader DataLoaderExample --do-optimize
```

> 📌 **Note:**
>
> - **Results Location**: All results generated by the commands are saved in the `siapy-cli/artifacts` directory.
> - **Consistency**: Use the same flags and arguments across all steps to ensure consistency with the selected model and data loader.
