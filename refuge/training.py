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
import os
import random
import re

import requests
import torch
import transformers
from tqdm import tqdm
from transformers import Adafactor, GPTNeoXTokenizerFast

from refuge._vendor.mkultra.soft_prompt import SoftPrompt
from refuge._vendor.mkultra.tuning import GPTNeoXPromptTuningLM

from . import config
from ._paths import TRAINING_DATA_DIR

Config = config.Config


def main(cfg: Config):
    training_data_path = TRAINING_DATA_DIR / "alice.txt"

    if not training_data_path.exists():
        _create_alice_txt(training_data_path)

    tokenizer = GPTNeoXTokenizerFast.from_pretrained(
        cfg.model.hugging_face_name, padding_side="left"
    )
    model = GPTNeoXPromptTuningLM.from_pretrained(
        cfg.model.hugging_face_name, device_map="auto"
    )

    model_base_name = config.get_model_base_name(cfg)
    project_checkpoint_dir = config.get_project_checkpoint_dir(cfg)

    filename_for_checkpoint = (
        lambda step: f"{sp_name}-{model_base_name}-step-{step}.json"
    )
    loaded_sp = None
    project_files = None

    # Look for existing checkpoints
    project_files = os.listdir(project_checkpoint_dir)
    if project_files is not None:
        checkpoint_files = [
            check_file for check_file in project_files if ("-step-" in check_file)
        ]

        if len(checkpoint_files) > 0:
            highest_step = max(
                [
                    int(check_file[check_file.rfind("-step-") + 6 : -5])
                    for check_file in checkpoint_files
                ]
            )
            loaded_sp = SoftPrompt.from_file(
                os.path.join(
                    project_checkpoint_dir, filename_for_checkpoint(highest_step)
                )
            )
            print(f"Loading latest checkpoint: {highest_step}")
        else:
            print("No checkpoints found")

    text_tokenized = None
    tokens_path = os.path.join(project_checkpoint_dir, "tokens.json")

    # See if we already have a tokens file
    try:
        with open(tokens_path, encoding="utf-8") as file:
            text_tokenized = json.load(file)
            print("Loaded existing tokens.json file")

    except FileNotFoundError:
        print("No tokens.json exists, creating it...")

    # If not, make one now
    if text_tokenized is None:
        with open(training_data_path, encoding="utf-8") as file:
            text = file.read()
        text_tokenized = tokenizer.encode(text)

        with open(tokens_path, "x", encoding="utf-8") as file:
            json.dump(text_tokenized, file)

    text_length = len(text_tokenized)
    num_blocks = math.ceil(text_length / block_size)

    print(f"Length of text: {len(text_tokenized)} tokens")
    print(f"Number of blocks: {num_blocks}, each {block_size} tokens")

    # Partition tokens into blocks
    blocks = list()
    for block_num in range(num_blocks):
        start = block_num * block_size
        end = min(start + block_size, text_length)
        blocks.append(text_tokenized[start:end])

    block_order_path = os.path.join(project_checkpoint_dir, "block_order.json")

    # See if we already have a block_order file
    try:
        with open(block_order_path, encoding="utf-8") as file:
            block_order = json.load(file)
            print("Loaded existing block_order.json file")

    except FileNotFoundError:
        print("No block_order.json exists, creating it...")
        block_order = [*range(num_blocks)]

        with open(block_order_path, "x", encoding="utf-8") as file:
            json.dump(block_order, file)

    if loaded_sp is None:
        initial_sp = SoftPrompt.from_string(initial_prompt, model, tokenizer)
        print(f"Initial prompt length: {len(initial_sp)}")
        model.set_soft_prompt(initial_sp)

        sp_step = 0
        eval_loss = 100
    else:
        model.set_soft_prompt(loaded_sp)
        sp_step = loaded_sp._metadata["step"]
        eval_loss = loaded_sp._metadata["loss"]

    num_training_steps = scheduler_params["num_training_steps"]

    optimizer_params["params"] = [model.get_soft_params()]
    optimizer = Adafactor(**optimizer_params)
    optimizer.state["step"] = sp_step

    scheduler_params["optimizer"] = optimizer
    scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        **scheduler_params
    )

    torch.cuda.empty_cache()
    loss_log_path = os.path.join(project_checkpoint_dir, "loss_log.csv")
    progress_bar = tqdm(total=num_training_steps)
    optimizer.state["step"] = sp_step
    evals_since_last_improvement = 0
    best_eval = float("inf")

    # Fix eval order
    eval_order = [*range(num_blocks)]
    # random.seed(1234)
    random.shuffle(eval_order)

    # Function for gradient accumulation scheduling
    def get_acc_steps(sp_step):
        if acc_doubling_rate != 0:
            return round(base_acc_steps * math.pow(2, (sp_step / acc_doubling_rate)))
        else:
            return base_acc_steps

    for _session_step in range(num_training_steps):
        model.train()

        acc_steps = get_acc_steps(sp_step)

        for i in range(acc_steps):
            idx = (sp_step * acc_steps + i) % num_blocks

            # Shuffle blocks every epoch
            if idx == 0:
                random.shuffle(block_order)
                with open(block_order_path, "w", encoding="utf-8") as file:
                    json.dump(block_order, file)

            block = blocks[block_order[idx]]

            input_ids = torch.LongTensor(block).unsqueeze(0).cuda().detach()

            # Forward pass and optimize
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            instant_loss = loss.item()
            if math.isnan(instant_loss):
                torch.cuda.empty_cache()
                raise KeyboardInterrupt

            # Discard tensor that was moved to GPU
            del input_ids
            torch.cuda.empty_cache()

        # Accumulate gradients
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        optimizer.zero_grad()

        if math.isnan(instant_loss):
            torch.cuda.empty_cache()
            raise KeyboardInterrupt

        # Evaluate model and plot loss
        if sp_step % eval_interval == 0:
            model.eval()
            torch.cuda.empty_cache()
            eval_loss = 0

            with torch.no_grad():
                for eval_step in range(eval_blocks):
                    block = blocks[eval_order[eval_step]]
                    input_ids = torch.LongTensor(block).unsqueeze(0).cuda().detach()
                    eval_loss += model(
                        input_ids=input_ids, labels=input_ids
                    ).loss.item()

                    # Discard tensor that was moved to GPU
                    del input_ids
                    torch.cuda.empty_cache()

            eval_loss /= eval_blocks

            with open(loss_log_path, "a", encoding="utf-8") as file:
                file.write(f"{sp_step},{eval_loss}\n")

            # Stop if loss has plateaued
            if plateau_steps != 0:
                if eval_loss < best_eval:
                    best_eval = eval_loss
                    evals_since_last_improvement = 0
                else:
                    evals_since_last_improvement += 1
                if evals_since_last_improvement > plateau_steps:
                    print(f"No improvement for {plateau_steps} evals")
                    break

        # Save checkpoint every so often
        if sp_step % checkpoint_interval == 0:
            sp = SoftPrompt.from_tuning_model(
                model,
                {
                    "name": sp_name + f"-step-{sp_step}",
                    "step": sp_step,
                    "loss": eval_loss,
                },
            )
            sp.to_file(
                os.path.join(project_checkpoint_dir, filename_for_checkpoint(sp_step))
            )

        progress_bar.set_postfix(
            {
                "Model Step": sp_step,
                "Eval Loss": "{el:.5f}".format(el=eval_loss),
                "Acc Steps": acc_steps,
                "LR": lr,
            }
        )
        progress_bar.update(1)
        sp_step += 1

    # Save a checkpoint once done
    sp = SoftPrompt.from_tuning_model(
        model,
        {"name": sp_name + f"-step-{sp_step}", "step": sp_step, "loss": eval_loss},
    )
    sp.to_file(os.path.join(project_checkpoint_dir, filename_for_checkpoint(sp_step)))


def _create_alice_txt(path):
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


def _regex_replace(s: str, regex, group, replacement):
    pat = re.compile(regex)
    while True:
        m = pat.search(s)
        if m is not None:
            s = s[: m.start(group)] + replacement + s[m.end(group) :]
        else:
            break
    return s
