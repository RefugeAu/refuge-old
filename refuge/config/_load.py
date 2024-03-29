# Copyright (C) Alex Carpenter and Simon Biggs

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


from types import SimpleNamespace

import tomlkit

from refuge._paths import CHECKPOINTS_DIR, LIB

DEFAULT = LIB / "config" / "default.toml"


class ProjectConfig(SimpleNamespace):
    name: str


class DataConfig(SimpleNamespace):
    name: str
    url: str
    start: str
    end: str


class ModelConfig(SimpleNamespace):
    hugging_face_name: str


class PromptConfig(SimpleNamespace):
    initializer: str


class TrainingConfig(SimpleNamespace):
    checkpoint_interval: int
    eval_interval: int
    eval_blocks: int
    base_acc_steps: int
    acc_doubling_rate: int
    plateau_steps: int
    block_size: int
    batch_size: int


class OptimizerConfig(SimpleNamespace):
    lr: float
    beta1: float
    decay_rate: float
    weight_decay: float
    scale_parameter: bool
    relative_step: bool


class SchedulerConfig(SimpleNamespace):
    num_warmup_steps: int
    num_cycles: int
    num_training_steps: int


class Config(SimpleNamespace):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    prompt: PromptConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.project = ProjectConfig(**kwargs["project"])
        self.data = DataConfig(**kwargs["data"])
        self.model = ModelConfig(**kwargs["model"])
        self.prompt = PromptConfig(**kwargs["prompt"])
        self.training = TrainingConfig(**kwargs["training"])
        self.optimizer = OptimizerConfig(**kwargs["optimizer"])
        self.scheduler = SchedulerConfig(**kwargs["scheduler"])


def load_config():
    with open(DEFAULT, encoding="utf-8") as f:
        cfg = Config(**tomlkit.load(f).unwrap())

    return cfg


def get_model_base_name(cfg: Config):
    return cfg.model.hugging_face_name.split("/")[-1]


def get_project_checkpoint_dir(cfg: Config):
    model_base_name = get_model_base_name(cfg)
    path = CHECKPOINTS_DIR / model_base_name / cfg.project.name
    path.mkdir(exist_ok=True, parents=True)

    return path
