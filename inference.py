import torch

from model import (
    GPTLanguageModel,
    n_embd,
    n_head,
    n_layer,
    block_size,
    dropout,
)

from tokenizer import (
    vocab_size,
    encode,
    decode,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GPTInference:
    def __init__(self, checkpoint_path="notebooks/checkpoints/gpt_model_final.pt"):

        self.device = DEVICE

        self.model = GPTLanguageModel(
            vocab_size=vocab_size,
            n_embd=n_embd,
            block_size=block_size,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint)

        self.model.eval()

    def generate(
        self,
        prompt="",
        max_new_tokens=200,
        temperature=1.0,
    ):

        if prompt.strip() == "":
            context = torch.zeros(
                (1, 1),
                dtype=torch.long,
                device=self.device,
            )
        else:
            context = torch.tensor(
                [encode(prompt)],
                dtype=torch.long,
                device=self.device,
            )

        with torch.no_grad():

            idx = context

            for _ in range(max_new_tokens):

                idx_cond = idx[:, -self.model.block_size:]

                logits, _ = self.model(idx_cond)

                logits = logits[:, -1, :]

                logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)

                idx_next = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                idx = torch.cat((idx, idx_next), dim=1)

        return decode(idx[0].tolist())