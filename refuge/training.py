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


import json
import math
import random
import re

import requests
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import Adafactor, GPTNeoXTokenizerFast

from . import config
from ._checkpoints import load_latest_checkpoint, save_checkpoint
from ._embeddings import get_embeddings
from ._paths import TRAINING_DATA_DIR
from .model import GPTNeoXPromptTuningLM

Config = config.Config


def train(cfg: Config, tokenizer: GPTNeoXTokenizerFast, model: GPTNeoXPromptTuningLM):
    try:
        step, model.soft_prompt_embeddings = load_latest_checkpoint(cfg)
    except FileNotFoundError:
        step = 0
        model.soft_prompt_embeddings = get_embeddings(
            tokenizer, model, cfg.prompt.initializer
        )

    text_tokenized = _get_tokenized_text(cfg, tokenizer)
    eval_split = cfg.training.block_size * cfg.training.eval_blocks
    training = text_tokenized[:eval_split]
    evaluation = text_tokenized[eval_split:]

    evaluation_blocks = [
        evaluation[i : i + cfg.training.block_size]
        for i in range(0, cfg.training.eval_blocks, cfg.training.block_size)
    ]

    parameters_to_train = [model.soft_prompt_parameter]
    optimizer = Adafactor(parameters_to_train, **cfg.optimizer.__dict__)
    optimizer.state["step"] = step

    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.scheduler.num_warmup_steps,
        num_training_steps=cfg.scheduler.num_training_steps,
        num_cycles=cfg.scheduler.num_cycles,
    )

    progress_bar = tqdm(total=cfg.scheduler.num_training_steps)

    try:
        _inner_loop(
            cfg=cfg,
            model=model,
            training=training,
            evaluation_blocks=evaluation_blocks,
            optimizer=optimizer,
            scheduler=scheduler,
            progress_bar=progress_bar,
        )
    finally:
        save_checkpoint(cfg, optimizer.state["step"], model.soft_prompt_embeddings)
        progress_bar.close()


def _inner_loop(
    cfg: Config,
    model: GPTNeoXPromptTuningLM,
    training: list[int],
    evaluation_blocks: list[list[int]],
    optimizer: Adafactor,
    scheduler: LambdaLR,
    progress_bar: tqdm,
):
    num_text_tokens = len(training)
    max_block_start = num_text_tokens - cfg.training.block_size

    eval_loss = torch.inf

    for i in range(cfg.scheduler.num_training_steps):
        model.train()

        acc_steps = _get_acc_steps(cfg, optimizer.state["step"])

        for _ in range(acc_steps):
            blocks = []
            for _ in range(cfg.training.batch_size):
                block_start = random.randint(0, max_block_start)
                block_end = block_start + cfg.training.block_size

                block = training[block_start:block_end]
                blocks.append(block)

            blocks_tensor = torch.LongTensor(blocks).to(model.device)
            outputs = _evaluate_model(model, blocks_tensor)

            loss = outputs.loss
            loss.backward()

        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        optimizer.zero_grad()

        optimizer.state["step"] += 1

        if optimizer.state["step"] % cfg.training.checkpoint_interval == 0:
            save_checkpoint(cfg, optimizer.state["step"], model.soft_prompt_embeddings)

        if optimizer.state["step"] % cfg.training.eval_interval == 1 or i == 0:
            model.eval()
            eval_loss = 0

            with torch.no_grad():
                for block in evaluation_blocks:
                    blocks_tensor = (
                        torch.LongTensor(block).unsqueeze(0).to(model.device)
                    )
                    outputs = _evaluate_model(model, blocks_tensor)

                    eval_loss += outputs.loss.item()

            eval_loss /= cfg.training.eval_blocks

        progress_bar.set_postfix(
            {
                "Model Step": optimizer.state["step"],
                "Eval Loss": f"{eval_loss:.5f}",
                "Acc Steps": acc_steps,
                "LR": lr,
            },
            refresh=False,
        )
        progress_bar.update(1)


def _evaluate_model(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    inputs_embeds = _cat_learned_embedding_to_input(model, blocks)
    labels = _extend_labels(model, blocks)

    outputs = model(
        inputs_embeds=inputs_embeds.to(model.device),
        labels=labels.to(model.device),
    )

    return outputs


def _get_acc_steps(cfg: Config, sp_step):
    if cfg.training.acc_doubling_rate != 0:
        return round(
            cfg.training.base_acc_steps
            * math.pow(2, (sp_step / cfg.training.acc_doubling_rate))
        )

    return cfg.training.base_acc_steps


def _get_tokenized_text(cfg: Config, tokenizer: GPTNeoXTokenizerFast) -> list[int]:
    model_base_name = config.get_model_base_name(cfg)
    path = TRAINING_DATA_DIR / f"alice-tokenized-{model_base_name}.json"

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        pass

    text = _get_raw_text()
    tokens = tokenizer.encode(text)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokens, f)

    return tokens


def _get_raw_text():
    path = TRAINING_DATA_DIR / "alice.txt"

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        return text
    except FileNotFoundError:
        pass

    data_str = requests.get(
        "https://www.gutenberg.org/files/11/11-0.txt", timeout=60
    ).content.decode("utf-8")
    clean_data_str = data_str

    clean_data_str = _regex_replace(clean_data_str, r"\r", 0, "")
    clean_data_str = _regex_replace(clean_data_str, r"\S(\n)\S", 1, " ")
    clean_data_str = _regex_replace(clean_data_str, r"\u201C", 0, '"')
    clean_data_str = _regex_replace(clean_data_str, r"\u201D", 0, '"')
    clean_data_str = _regex_replace(clean_data_str, r"_", 0, "")
    clean_data_str = clean_data_str[1434:-18595]

    with open(path, "w", encoding="utf-8") as f:
        f.write(clean_data_str)

    return clean_data_str


def _regex_replace(s: str, regex, group, replacement):
    pat = re.compile(regex)
    while True:
        m = pat.search(s)
        if m is not None:
            s = s[: m.start(group)] + replacement + s[m.end(group) :]
        else:
            break
    return s


def _cat_learned_embedding_to_input(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    base_embeddings = model.get_token_embeddings(blocks)

    # TODO: Handle this directly with the get_token_embeddings function
    inputs_embeds = torch.cat(
        [
            model.soft_prompt_parameter.repeat(base_embeddings.size(0), 1, 1),
            base_embeddings,
        ],
        dim=1,
    )

    return inputs_embeds


def _extend_labels(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    n_tokens = model.soft_prompt_parameter.shape[-2]

    # TODO: Clean this up
    # Add '-100's (prevent loss calculation where the learned embed would be)
    n_batches = blocks.shape[0]
    return torch.cat(
        [torch.full((n_batches, n_tokens), -100, device=model.device), blocks], dim=1
    )
