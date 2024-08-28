import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = ".env"

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SAVED_DIR = BASE_DIR / "artifacts"

SAVED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True, dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    project_name: str = Field(
        default=None,
        description="Project name - artifacts saved under this name in 'saved' dictionary",
    )
    images_dir: Path = Field(
        default=None, description="Path to spectral images directory."
    )
    artifacts_dir: Path = Field(
        default=SAVED_DIR,
        description="Path to program artifacts directory. Currently cannot be changed.",
    )
    debug: bool = Field(
        default=False, description="If logging displays debug information."
    )
    header_file_suffix: str = ".hdr"
    image_file_suffix: str = ".img"
    camera1_id: str = "VNIR_1600_SN0034"
    camera2_id: str = "SWIR_384me_SN3109"
    labels_part_deliminator: str = "__"
    labels_between_deliminator: str = "_"

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        cli_parse_args=False,
        env_ignore_empty=True,
    )

    @field_validator("images_dir", mode="before")
    @classmethod
    def correct_path(cls, path: str | Path):
        if path is None:
            raise ValueError("Define 'IMAGES_DIR' in .env file.")
        path = str(path)
        # If the os is run as wsl (windows subsystem for linux)
        if ":\\" in path:
            path = re.sub(
                r"([A-Za-z]):\\", lambda m: f"/mnt/{m.group(1).lower()}/", path
            ).replace("\\", "/")
            if path.startswith('r"') and path.endswith('"'):
                path = path[2:-1]
        path = Path(path)
        if not path.is_dir():
            raise ValueError(
                f"Ensure defined 'IMAGES_DIR' is a directory. Current value: {path}."
            )
        return path

    @field_validator("project_name", mode="before")
    @classmethod
    def validate_project_name(cls, project_name: str):
        if project_name is None:
            raise ValueError("Define 'PROJECT_NAME' in .env file.")
        return project_name


settings = Settings()
