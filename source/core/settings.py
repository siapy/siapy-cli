from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DOTENV_PATH = ".env"

BASE_DIR = Path(__file__).parent.parent.parent.absolute()
SAVED_DIR = BASE_DIR / "saved"

SAVED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(override=True, dotenv_path=DOTENV_PATH)


class Settings(BaseSettings):
    images_dir: Path = Field(
        default=None, description="Path to spectral images directory."
    )
    debug: bool = Field(
        default=False, description="If logging displays debug information."
    )

    model_config = SettingsConfigDict(
        env_file=DOTENV_PATH,
        env_file_encoding="utf-8",
        cli_parse_args=False,
        env_ignore_empty=True,
    )

    @field_validator("images_dir", mode="before")
    @classmethod
    def correct_path(cls, path):
        if path is None:
            raise ValueError("Define 'IMAGES_DIR' in .env file.")
        if ":\\" in str(path):
            path = str(path).replace("F:\\", "/mnt/f/").replace("\\", "/")
            if path.startswith('r"') and path.endswith('"'):
                path = path[2:-1]
        path = Path(path)
        if not path.is_dir():
            raise ValueError(
                f"Ensure defined 'IMAGES_DIR' is a directory. Current value: {path}."
            )
        return path


settings = Settings()
