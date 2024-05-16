"""
Download data from a drive link and put it in the data/raw directory.
"""

import gdown
import tyro
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union
from typing_extensions import Annotated
from nerfstudio.utils.rich_utils import CONSOLE


def download_file(url, output_path):
    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the output file and write the content
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Download completed: {output_path}")
    else:
        print(f"Error: Unable to download the file. HTTP Status Code: {response.status_code}")


@dataclass
class DataDownload:
    """Download a dataset"""

    save_dir: Path = Path("data/")
    """The directory to save the dataset to"""

    def download(self, save_dir: Path) -> None:
        """Download the dataset"""
        raise NotImplementedError


images_dict = {
    "bobs-burgers-dining": "https://drive.google.com/drive/folders/1eicpYlQRUMuCbWfBG5SZev_4Y8FKHMKS",
    "family-guy-house": "https://drive.google.com/drive/folders/1cJ6NSDpTn620S3mceKOe1jNlcyb7szpS",
    "smith-dining": "https://drive.google.com/drive/folders/1vILY-3h3-IGqX4sR-1OFQQxHH3GpgWUu",
    "smith-kitchen": "https://drive.google.com/drive/folders/1t8Vu6iqEr0MnyYWlt3DYQRqcRzheeieY",
    "smith-residence": "https://drive.google.com/drive/folders/1bRcHjEpdftOh_131tqGY6abyLpKgz-3S",
    "kora-fight": "https://drive.google.com/drive/folders/1Dxi__Wg_E1x9K9m67kIwZNxHDBsUB27l",
    "scott-bedroom": "https://drive.google.com/drive/folders/1Y7oXrHZfxQAZcH-5kCSQjl2WSnVra3CK",
    "planet-express": "https://drive.google.com/drive/folders/1bH8dSRBtJ-ihHaKdRajFF0Knexh5i3bj",
    "family-guy-dining": "https://drive.google.com/drive/folders/1yi6pKAY1CUP_5QFx8UxchgSCIgiEyJhk",
    "phineas-ferb-house": "https://drive.google.com/drive/folders/1WV_GwRhc7EyzHEoPZRrhcnnKMum-HEgJ",
    "spirited-away-train": "https://drive.google.com/drive/folders/1DuVg9KVNNwAAtbSi9dWgp-TEk8BL9LQj",
}


@dataclass
class ImagesDownload(DataDownload):
    """Download a folder of images. If dataset is None, we print the options."""

    dataset: Optional[str] = None
    """Dataset name"""
    link: Optional[str] = None
    """The link to download from"""

    def download(self, save_dir: Path):
        """Download a folder of images."""
        if self.dataset is None:
            CONSOLE.print("[bold yellow]Please specify a dataset to download from the following options with --dataset:")
            for k in images_dict:
                CONSOLE.print(f"[bold green]    {k}")
            return
        if self.link is not None and self.dataset in images_dict:
            print("Overriding dataset link with provided link.")
        if self.link is None:
            if self.dataset not in images_dict:
                raise ValueError(f"Dataset {self.dataset} not found. Please specify a link.")
            self.link = images_dict[self.dataset]
        download_path = save_dir / Path("images") / self.dataset
        gdown.download_folder(url=self.link, output=str(download_path), quiet=False, use_cookies=False)


dataset_dict = {
    "smith-kitchen": "https://drive.google.com/drive/folders/1EPXWRFwjRiAmlAQ8sKFmeGMpEjo9-BY4",
    "scott-bedroom": "https://drive.google.com/drive/folders/1qPoWFxGBrl-yo5Apn9cWpf0EDwQQpE43",
    "smith-residence": "https://drive.google.com/drive/folders/1uTOnWj3otNo5qIp-a1Y0Z9kLxG1HNtf6",
    "family-guy-house": "https://drive.google.com/drive/folders/1dfTgbz1nfKFxM4GeN4Pf_uK0qjfIRHpr",
    "smith-dining": "https://drive.google.com/drive/folders/1DoVYltclyoPGlys3IRkBwBO8kO0QxvbH",
    "bobs-burgers-dining": "https://drive.google.com/drive/folders/19coGtXgvQa7FrTCa5fXcip3tutXfH2S1",
}


@dataclass
class DatasetDownload(DataDownload):
    """Download a preprocessed Nerfstudio dataset.
    This is a Nerfstudio input."""

    dataset: Optional[str] = None
    """Dataset name"""
    link: Optional[str] = None
    """The link to download from"""

    def download(self, save_dir: Path):
        """Download a dataset."""
        if self.dataset == "all":
            for k, v in dataset_dict.items():
                self.dataset = k
                self.link = v
                self.download(save_dir)
            return
        if self.dataset is None:
            CONSOLE.print("[bold yellow]Please specify a dataset to download from the following options with --dataset:")
            for k in dataset_dict:
                CONSOLE.print(f"[bold green]    {k}")
            return
        if self.link is not None and self.dataset in dataset_dict:
            print("Overriding dataset link with provided link.")
        if self.link is None:
            if self.dataset not in dataset_dict:
                raise ValueError(f"Dataset {self.dataset} not found. Please specify a link.")
            self.link = dataset_dict[self.dataset]
        download_path = save_dir / Path("processed") / self.dataset
        gdown.download_folder(url=self.link, output=str(download_path), quiet=False, use_cookies=False)


@dataclass
class SamDownload(DataDownload):
    """Download segment anything weights."""

    def download(self, save_dir: Path) -> None:
        """Download segment anything weights."""
        download_path = save_dir / Path("sam-checkpoints") / "sam_vit_b_01ec64.pth"
        download_path.parent.mkdir(parents=True, exist_ok=True)
        CONSOLE.print(f"Downloading SAM weights to {download_path}")
        download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth", download_path)


Commands = Union[
    Annotated[ImagesDownload, tyro.conf.subcommand(name="images")],
    Annotated[DatasetDownload, tyro.conf.subcommand(name="dataset")],
    Annotated[SamDownload, tyro.conf.subcommand(name="sam")],
]


def main(
    data_download: DataDownload,
):
    """Script to download data.
    - images: Blender synthetic scenes realeased with NeRF.
    - meshes: Meshes to apply NeRFiller to.

    Args:
        dataset: The dataset to download (from).
    """
    data_download.save_dir.mkdir(parents=True, exist_ok=True)

    data_download.download(data_download.save_dir)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    main(tyro.cli(Commands))


if __name__ == "__main__":
    entrypoint()
