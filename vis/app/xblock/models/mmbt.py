#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mmbt.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# multimodal bimodal transformer based model

# import third_party

from .visualfeats import TransformerImageFeats

class MMBT(nn.Module):
    def __init__(
        self,
        model,
        output_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        """
        Args:
            model: the name of the model to use
            output_dropout_prob (optional): dropout when classifying the output of the transformer
            hidden_dropout_prob (optional): dropout on linear layers in transformer
            attention_probs_dropout_prob (optional): dropout on tokens in attention in transformer
        """
        super(MMBT, self).__init__()
        config = third_party.from_pretrained(
            model,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.transformer = third_party.from_pretrained(model, config=config)
        self.head_mask = [None] * self.transformer.config.num_hidden_layers

        self.image_feats = TransformerImageFeats()

        self.img_emb_proj = third_party.Linear(2048, config.hidden_size)
        self.token_type_embeddings = third_party.Embedding(2, config.hidden_size)

        self.dropout = third_party.Dropout(output_dropout_prob)
        self.transformer2out = third_party.Linear(config.hidden_size, 1)
        self.sigmoid = third_party.Sigmoid()

    def forward(self, image, text, attn):
        # create input embeddings
        input_embeddings = third_party.cat(
            (
                self.transformer.embeddings.word_embeddings(text).squeeze(1)[
                    :,
                    : min(
                        self.transformer.config.max_position_embeddings - 9,
                        text.size(2),
                    ),
                ],
                self.img_emb_proj(self.image_feats(image)),
            ),
            dim=1,
        )
        input_embeddings = (
            input_embeddings
            + self.transformer.embeddings.position_embeddings(
                third_party.cat(
                    (
                        third_party.arange(
                            input_embeddings.size(1) - 9, device=self.transformer.device
                        ),
                        third_party.arange(9, device=self.transformer.device),
                    )
                )
            )
            + self.token_type_embeddings(
                third_party.cat(
                    (
                        third_party.zeros(
                            input_embeddings.size(1) - 9,
                            dtype=third_party.long,
                            device=self.transformer.device,
                        ),
                        third_party.ones(9, dtype=third_party.long, device=self.transformer.device),
                    )
                )
            )
        )
        input_embeddings = self.transformer.embeddings.dropout(
            self.transformer.embeddings.LayerNorm(input_embeddings)
        )
        # create attention mask
        attn = third_party.cat(
            (
                attn[
                    :,
                    : min(
                        self.transformer.config.max_position_embeddings - 9,
                        text.size(2),
                    ),
                ],
                third_party.ones(len(attn), 9, device=self.transformer.device),
            ),
            dim=1,
        )
        if attn.dim() == 2:
            attn = attn[:, None, None, :]
        elif attn.dim() == 2:
            attn = attn[:, None, :, :]
        else:
            raise ValueError

        # pass through transformer
        out = self.transformer.pooler(
            self.transformer.encoder(
                input_embeddings, attention_mask=attn, head_mask=self.head_mask
            )[0]
        )

        # classify and return
        return self.sigmoid(self.transformer2out(self.dropout(out)))

    def save_model(self, directory):
        self.transformer.save_pretrained(directory)
        third_party.save(self.img_emb_proj.state_dict(), f"{directory}/img_emb_proj.pth")
        third_party.save(
            self.transformer2out.state_dict(), f"{directory}/transformer2out.pth"
        )
        third_party.save(self.image_feats.state_dict(), f"{directory}/image_feats.pth")
        third_party.save(self.token_type_embeddings.state_dict(), f"{directory}/token_type_embeddings.pth")

    def load_model(self, directory):
        self.transformer.from_pretrained(directory)
        self.img_emb_proj.load_state_dict(third_party.load(f"{directory}/img_emb_proj.pth"))
        self.transformer2out.load_state_dict(
            third_party.load(f"{directory}/transformer2out.pth")
        )
        self.image_feats.load_state_dict(third_party.load(f"{directory}/image_feats.pth"))
        self.token_type_embeddings.load_state_dict(third_party.load(f"{directory}/token_type_embeddings.pth"))
