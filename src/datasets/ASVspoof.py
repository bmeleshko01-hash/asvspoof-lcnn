from pathlib import Path
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVspoofDataset(BaseDataset):
    """
    ASVspoof 2019 dataset.

    Reads metadata from an ASVspoof protocol file.
    """

    def __init__(
        self,
        audio_dir,
        protocol_file,
        name="train",
        *args,
        **kwargs,
    ):
        """
        Args:
            audio_dir (str): directory containing audio files.
            protocol_file (str): path to the protocol file.
            name (str): partition name: train, dev or eval.
        """
        self.audio_dir = Path(audio_dir)
        self.protocol_file = Path(protocol_file)

        index_path = (
            ROOT_PATH
            / "data"
            / "asvspoof"
            / "indexes"
            / name
            / "index.json"
        )

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(index_path, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, index_path, name):
        """
        Create index for the dataset.

        Args:
            index_path (Path): path where index.json will be saved.
            name (str): partition name: train, dev or eval.

        Returns:
            index (list[dict]): list containing metadata for each
                dataset object.
        """
        index = []
        index_path.parent.mkdir(exist_ok=True, parents=True)

        with self.protocol_file.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        print(f"Parsing ASVspoof Dataset metadata for part {name}...")

        for line in tqdm(lines):
            parts = line.strip().split()

            audio_id = parts[1]
            label_name = parts[4]

            if label_name == "bonafide":
                label = 1
            else:
                label = 0

            audio_path = Path(self.audio_dir) / f"{audio_id}.flac"

            index.append(
                {
                    "path": str(audio_path),
                    "label": label,
                }
            )
        write_json(index, str(index_path))

        return index

    def load_object(self, path):
        waveform, sample_rate = torchaudio.load(path)
        return waveform