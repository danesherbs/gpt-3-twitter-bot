from dataclasses import dataclass
import math
import torch
from torchtyping import TensorType
import einops
import transformers


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "vocab_size"]
    final_encoding: TensorType["batch_size", "hidden_size"]


class MultiheadAttention(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads
        self.linear_attn = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.linear_output = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: TensorType["batch_size", "seq_len", "hidden_size"]):
        batch_size, seq_len, hidden_size = x.shape
        attn_concat = self.linear_attn(x)

        q, k, v = torch.split(attn_concat, self.hidden_size, dim=-1)
        q = einops.rearrange(q, "b n (h l) -> b h n l", l=self.head_size)
        k = einops.rearrange(k, "b n (h l) -> b h n l", l=self.head_size)
        v = einops.rearrange(v, "b n (h l) -> b h n l", l=self.head_size)

        attn_raw = torch.einsum("bhts, bhfs -> bhtf", q, k)

        neg_inf = torch.tensor(-1e4)
        attn_mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1
        )
        attn_masked = torch.where(attn_mask, neg_inf, attn_raw) / math.sqrt(
            self.head_size
        )
        attn_scores = torch.softmax(attn_masked, dim=-1)

        attn = torch.einsum("bhtf, bhfs -> bhts", attn_scores, v)
        attn = einops.rearrange(attn, "b h n l -> b n (h l)")

        return self.linear_output(attn)


class GPT2Block(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
    ) -> None:
        super().__init__()
        self.norm_1 = torch.nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = MultiheadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.norm_2 = torch.nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear_1 = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.linear_2 = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: TensorType["batch_size", "seq_len", "hidden_size"]):
        normed_1 = self.norm_1(x)
        attn = self.attn(normed_1)
        attn_resid = x + attn
        normed_2 = self.norm_2(attn_resid)
        linear_1 = self.linear_1(normed_2)
        gelu = torch.nn.functional.gelu(linear_1)
        linear_2 = self.linear_2(gelu)
        dropout = self.dropout(linear_2)
        return dropout + attn_resid


class GPT2(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        dropout,
        layer_norm_epsilon,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.pos_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.gpt_blocks = torch.nn.ModuleList(
            [
                GPT2Block(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")

    def forward(self, input_ids: TensorType["batch_size", "seq_len"]):
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        x = self.dropout(x)

        for block in self.gpt_blocks:
            x = block(x)

        x = self.norm(x)
        final_encoding = x[:, -1]
        logits = final_encoding @ self.token_embeddings.weight.T

        return GPT2Output(
            logits=logits,
            final_encoding=final_encoding,
        )

    def next_token(
        self,
        input_ids: TensorType["seq_len"],
        temperature: float,
        freq_penalty: float = 2.0,
    ) -> TensorType["vocab_size"]:
        output = self.forward(input_ids)
        id_frequencies = torch.bincount(
            input_ids.squeeze(0), minlength=self.vocab_size
        ).unsqueeze(0)
        return torch.nn.functional.softmax(
            output.logits / temperature - id_frequencies * freq_penalty
        )

    def generate(
        self,
        text: str,
        max_length: int = 30,
        temperature: float = 1.0,
        freq_penalty: float = 2.0,
    ):
        output_ids = torch.tensor([self.tokenizer(text).input_ids])
        seq_len = output_ids.shape[1]

        while seq_len < max_length:
            output_logits = self.next_token(output_ids, temperature, freq_penalty)
            next_token_id = torch.argmax(output_logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)
            seq_len = output_ids.shape[1]

            if next_token_id == self.tokenizer.eos_token_id:
                break  # reached end of sentence

        return self.tokenizer.batch_decode(output_ids)


my_gpt = GPT2(
    num_layers=12,
    num_heads=12,
    vocab_size=50257,
    hidden_size=768,
    max_position_embeddings=1024,
    dropout=0.1,
    layer_norm_epsilon=1e-5,
)

my_gpt.load_state_dict(torch.load("/my_gpt.pt"))
