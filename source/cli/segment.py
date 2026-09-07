import typer
from source.misc.display_image import (
    display_spectral_images_with_areas,
)
from source.segmentation import (
    convert_images_to_reflectance,
    convert_selected_areas_to_train_data,
    create_spectral_signatures,
    find_transformation_between_images,
    perform_segmentation,
    select_areas_on_images,
    train_xgboost_model,
)
from source.segmentation.artifacts import (
    load_all_selected_areas,
    load_model,
    load_transformation_matrix,
    save_model,
    save_selected_areas,
    save_transformation_matrix,
)

app = typer.Typer()


@app.command()
def calculate_transformation(label: str):
    matx = find_transformation_between_images(label)
    save_transformation_matrix(matx)


@app.command()
def select_areas(label: str, category: str):
    selected_areas = select_areas_on_images(label)
    matx = load_transformation_matrix()
    display_spectral_images_with_areas(label, selected_areas, matx)
    save_selected_areas(selected_areas, category, label)


@app.command()
def train_model():
    selected_areas = load_all_selected_areas()
    matx = load_transformation_matrix()
    X_cam1, y_cam1, X_cam2, y_cam2 = convert_selected_areas_to_train_data(
        selected_areas, matx
    )
    encoder_cam1, model_cam1 = train_xgboost_model(X_cam1, y_cam1)
    encoder_cam2, model_cam2 = train_xgboost_model(X_cam2, y_cam2)
    save_model(encoder_cam1, model_cam1, encoder_cam2, model_cam2)


@app.command()
def segment_images(label: str | None = None):
    encoder_cam1, model_cam1, encoder_cam2, model_cam2 = load_model()
    matx = load_transformation_matrix()
    perform_segmentation(
        encoder_cam1,
        model_cam1,
        encoder_cam2,
        model_cam2,
        matx,
        label,
    )


@app.command()
def convert_to_reflectance(panel_reflectance: float):
    convert_images_to_reflectance(panel_reflectance)


@app.command()
def create_signatures(average_pixels: bool = True):
    create_spectral_signatures(average_pixels)
