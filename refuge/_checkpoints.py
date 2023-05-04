# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pathlib

import pandas as pd
import torch

from . import config

Config = config.Config


def load_latest_checkpoint(cfg: Config):
    steps_dir = _get_steps_dir(cfg)
    checkpoint_ids = [int(item.stem) for item in steps_dir.glob("*.csv")]

    if not checkpoint_ids:
        raise FileNotFoundError("No checkpoints found")

    latest_checkpoint_id = max(checkpoint_ids)

    path = steps_dir / f"{latest_checkpoint_id}.csv"

    print(f"Loading checkpoint from {path}")

    return latest_checkpoint_id, _load_embeddings_from_file(path)


def save_checkpoint(cfg: Config, step, embeddings: torch.Tensor):
    steps_dir = _get_steps_dir(cfg)
    path = steps_dir / f"{step}.csv"

    _save_embeddings_to_file(embeddings, path)


def _load_embeddings_from_file(path: pathlib.Path):
    return torch.tensor(
        pd.read_csv(path, index_col=None, header=None).values, dtype=torch.float32
    )


def _save_embeddings_to_file(embeddings: torch.Tensor, path: pathlib.Path):
    pd.DataFrame(embeddings.cpu().numpy()).to_csv(path, header=None, index=None)  # type: ignore


def _get_steps_dir(cfg: Config):
    path = config.get_project_checkpoint_dir(cfg)
    path.mkdir(exist_ok=True, parents=True)

    return path
