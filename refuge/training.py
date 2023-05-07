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
from typing import cast

import pytorch_optimizer
import requests
import torch
import transformers
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import Adafactor, AddedToken, GPTNeoXTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from . import config
from ._checkpoints import load_latest_checkpoint, save_checkpoint
from ._embeddings import get_embeddings
from ._gsam import GSAM
from ._paths import DATA_DIR
from .model import GPTNeoXPromptTuningLM

Config = config.Config


def get_tokenizer_and_model(cfg: Config):
    tokenizer: GPTNeoXTokenizerFast = GPTNeoXTokenizerFast.from_pretrained(
        cfg.model.hugging_face_name, padding_side="left"
    )
    model: GPTNeoXPromptTuningLM = GPTNeoXPromptTuningLM.from_pretrained(
        cfg.model.hugging_face_name, device_map="auto"
    )

    try:
        step, model.soft_prompt_embeddings = load_latest_checkpoint(cfg)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        step = 0
        model.soft_prompt_embeddings = get_embeddings(
            tokenizer, model, cfg.prompt.initializer
        )

    model.step = step

    number_of_soft_tokens = model.soft_prompt_embeddings.shape[0]
    soft_tokens = [f"<|{i}|>" for i in range(number_of_soft_tokens)]
    assert set(tokenizer.vocab.keys()).isdisjoint(soft_tokens)

    tokenizer.add_tokens(cast(list[str | AddedToken], soft_tokens))
    model.soft_tokens = soft_tokens

    return tokenizer, model


def train(
    cfg: Config,
    tokenizer: GPTNeoXTokenizerFast,
    model: GPTNeoXPromptTuningLM,
):
    # text_tokenized = _get_tokenized_text(cfg, tokenizer)
    # eval_split = cfg.training.block_size * cfg.training.eval_blocks
    # training = text_tokenized[:-eval_split]
    # evaluation = text_tokenized[-eval_split:]

    # print(len(training), len(evaluation))

    # evaluation_blocks = [
    #     evaluation[i : i + cfg.training.block_size]
    #     for i in range(
    #         0,
    #         cfg.training.eval_blocks * cfg.training.block_size,
    #         cfg.training.block_size,
    #     )
    # ]

    parameters_to_train = [model.soft_prompt_parameter]

    base_optimizer = torch.optim.AdamW(parameters_to_train)
    lr_scheduler = pytorch_optimizer.LinearScheduler(
        base_optimizer,
        max_lr=cfg.optimizer.lr,
        t_max=cfg.scheduler.num_training_steps,
        warmup_steps=cfg.scheduler.num_warmup_steps,
    )
    rho_scheduler = pytorch_optimizer.ProportionScheduler(
        lr_scheduler, max_lr=cfg.optimizer.lr, max_value=1, min_value=0.1
    )
    optimizer = GSAM(
        parameters_to_train, base_optimizer, model, rho_scheduler, adaptive=True
    )

    # scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=cfg.scheduler.num_warmup_steps,
    #     num_training_steps=cfg.scheduler.num_training_steps,
    #     num_cycles=cfg.scheduler.num_cycles,
    # )

    progress_bar = tqdm(total=cfg.scheduler.num_training_steps)

    try:
        _inner_loop(
            cfg=cfg,
            model=model,
            # training=training,
            # evaluation_blocks=evaluation_blocks,
            optimizer=optimizer,
            # scheduler=scheduler,
            lr_scheduler=lr_scheduler,
            progress_bar=progress_bar,
            tokenizer=tokenizer,
        )
    finally:
        save_checkpoint(cfg, model.step, model.soft_prompt_embeddings)
        progress_bar.close()


def _inner_loop(
    cfg: Config,
    tokenizer: GPTNeoXTokenizerFast,
    model: GPTNeoXPromptTuningLM,
    # training: list[int],
    # evaluation_blocks: list[list[int]],
    optimizer: pytorch_optimizer.GSAM,
    lr_scheduler,
    # scheduler: LambdaLR,
    progress_bar: tqdm,
):
    # num_text_tokens = len(training)
    # max_block_start = num_text_tokens - cfg.training.block_size

    # eval_loss = torch.inf

    for i in range(cfg.scheduler.num_training_steps):
        accumulated_loss = 0
        model.train()

        acc_steps = _get_acc_steps(cfg, model.step)

        accumulation_blocks = []

        for i in range(acc_steps):
            # if i == 0:
            #     model.run_soft_prompt_loss = True
            # else:
            #     model.run_soft_prompt_loss = False

            blocks = []
            for _ in range(cfg.training.batch_size):
                a = random.randint(1, 99)
                b = random.randint(1, 99)
                c = a + b

                block = tokenizer.encode(f"{model.soft_prompt}; {a} + {b} = {c};")

                blocks.append(block)

            accumulation_blocks.append(blocks)

        optimizer.forward_backward_func = lambda: gsam_forwards_and_backwards(
            optimizer, model, accumulation_blocks
        )
        _predictions, accumulated_loss = optimizer.step()
        lr_scheduler.step()
        optimizer.update_rho_t()

        # optimizer.base_optimizer.step()

        # optimizer.step()
        lr = lr_scheduler.get_lr()
        rho = optimizer.rho_t
        # scheduler.step()
        # optimizer.zero_grad()

        model.step += 1

        print(tokenizer.decode(model.translated_soft_prompt()).strip())

        if model.step % cfg.training.checkpoint_interval == 0:
            save_checkpoint(cfg, model.step, model.soft_prompt_embeddings)

        # if model.step % cfg.training.eval_interval == 1 or i == 0:
        #     model.eval()
        #     eval_loss = 0

        #     with torch.no_grad():
        #         for block in evaluation_blocks:
        #             blocks_tensor = (
        #                 torch.LongTensor(block).unsqueeze(0).to(model.device)
        #             )
        #             outputs = _evaluate_model(model, blocks_tensor)

        #             loss = outputs.loss
        #             assert loss is not None
        #             eval_loss += loss.item()

        #     eval_loss /= cfg.training.eval_blocks

        progress_bar.set_postfix(
            {
                "Model Step": model.step,
                "Loss": f"{accumulated_loss / cfg.training.base_acc_steps:.5f}",
                # "Eval Loss": f"{eval_loss:.5f}",
                "Acc Steps": acc_steps,
                "LR": lr,
                "Rho": rho,
            },
            refresh=False,
        )
        progress_bar.update(1)


def _get_inputs_and_targets(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    inputs = _cat_learned_embedding_to_input(model, blocks)
    targets = _extend_labels(model, blocks)

    return inputs.to(model.device), targets.to(model.device)


def _get_acc_steps(cfg: Config, sp_step):
    if cfg.training.acc_doubling_rate != 0:
        return round(
            cfg.training.base_acc_steps
            * math.pow(2, (sp_step / cfg.training.acc_doubling_rate))
        )

    return cfg.training.base_acc_steps


def gsam_forwards_and_backwards(optimizer, model, accumulation_blocks):
    optimizer.base_optimizer.zero_grad()

    with torch.enable_grad():
        loss = None
        for blocks in accumulation_blocks:
            blocks_tensor = torch.LongTensor(blocks).to(model.device)
            inputs, targets = _get_inputs_and_targets(model, blocks_tensor)

            outputs: CausalLMOutputWithPast = model(
                inputs_embeds=inputs,
                labels=targets,
            )

            if loss is None:
                loss = outputs.loss
            else:
                loss += outputs.loss

    loss.backward()

    return None, loss.detach()


# def _get_tokenized_text(cfg: Config, tokenizer: GPTNeoXTokenizerFast) -> list[int]:
#     model_base_name = config.get_model_base_name(cfg)
#     path = DATA_DIR / f"{cfg.data.name}-tokenized-{model_base_name}.json"

#     try:
#         with open(path, encoding="utf-8") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         pass

#     text = _get_raw_text(cfg)
#     tokens = tokenizer.encode(text)

#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(tokens, f)

#     return tokens


def _get_raw_text(cfg: Config):
    path = DATA_DIR / f"{cfg.data.name}.txt"

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()

        return text
    except FileNotFoundError:
        pass

    data_str = requests.get(cfg.data.url, timeout=60).content.decode("utf-8")

    start_index = data_str.index(cfg.data.start)
    end_index = data_str.index(cfg.data.end) + len(cfg.data.end)

    inner_data = data_str[start_index:end_index]
    inner_data = inner_data.strip() + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(inner_data)

    return inner_data


def _cat_learned_embedding_to_input(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    embeddings = model.convert_tokens_to_embeddings(blocks, trainable=True)

    return embeddings


def _extend_labels(model: GPTNeoXPromptTuningLM, blocks: torch.Tensor):
    soft_prompt_mask = blocks >= model.config.vocab_size
    masked_blocks = blocks.clone()

    # Prevents loss calculation on the soft prompt tokens
    masked_blocks[soft_prompt_mask] = -100

    return masked_blocks
