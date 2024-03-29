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

from typing import cast

from torch import nn
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast


def get_embeddings(
    tokenizer: GPTNeoXTokenizerFast, model: GPTNeoXForCausalLM, prompt: str
):
    tokens = tokenizer.encode(prompt)

    embedding_layer = cast(nn.Embedding, model.gpt_neox.embed_in)
    embeddings = embedding_layer.weight[tokens, :].clone().detach()

    return embeddings
