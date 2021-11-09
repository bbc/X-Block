#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : multitask.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>


# this code is the fundamental codepiece that supports multistream and
# multitask models

# import third_party



class MultitaskModel(nn.Module):
    def __init__(
        self,
        transformer,
        feature_dim=200,
        combine_dim=800,
        output_dims=[1, 1, 1],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        projection_dropout=0.1,
        large_cnn=False,
    ):
        super(MultitaskModel, self).__init__()

        config = third_party.from_pretrained(
            transformer,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.transformer = third_party.from_pretrained(transformer)
        self.image_cnn = (
            third_party.Sequential(*list(resnet152(pretrained=True).children())[:-1])
            if large_cnn
            else third_party.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        )

        self.transformer_hidden_size = self.transformer.config.hidden_size
        self.comment_proj = third_party.Linear(self.transformer_hidden_size, 200)
        self.title_proj = third_party.Linear(self.transformer_hidden_size, 200)
        self.image_text_proj = third_party.Linear(self.transformer_hidden_size, 200)
        self.image_proj = third_party.Linear(2048, 200) if large_cnn else third_party.Linear(512, 200)
        self.feature_dropout = nn.Dropout(projection_dropout)

        self.comment_layernorm = third_party.LayerNorm(feature_dim)
        self.title_layernorm = third_party.LayerNorm(feature_dim)
        self.image_layernorm = third_party.LayerNorm(feature_dim)
        self.image_text_layernorm = third_party.LayerNorm(feature_dim)

        self.projection = third_party.Sequential(
            third_party.Dropout(projection_dropout),
            third_party.ReLU(),
            third_party.Linear(feature_dim * 4, combine_dim),
            third_party.Dropout(projection_dropout),
            third_party.ReLU(),
            third_party.Linear(combine_dim, combine_dim),
            third_party.Dropout(projection_dropout),
            third_party.ReLU(),
        )
        self.output_dims = output_dims
        self.task_outputs = third_party.ModuleList(
            [third_party.Linear(combine_dim, i) for i in output_dims]
        )
        self.sigmoid = third_party.Sigmoid()
        self.feature_dim = feature_dim

    def forward(
        self,
        task_indices,
        comment=None,
        comment_attn=None,
        title=None,
        title_attn=None,
        image_text=None,
        image_text_attn=None,
        image=None,
        modalities=None,
    ):
        """
        Args:
            task_indices: the task that each example corresponds to
            modalities: give the indices that are avaliable at each data point
        """

        comment_feats = third_party.zeros(len(task_indices), self.feature_dim).to(
            self.transformer.device
        )
        if comment_feats is not None and (modalities[:, 0] == 1).sum().item():
            transformer_output = self.transformer(
                input_ids=comment[modalities[:, 0] == 1],
                attention_mask=comment_attn[modalities[:, 0] == 1],
            )[-1]
            comment_feats[modalities[:, 0] == 1] = self.comment_layernorm(
                self.comment_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )
        title_feats = third_party.zeros_like(comment_feats)
        if title is not None and (modalities[:, 1] == 1).sum().item():
            transformer_output = self.transformer(
                input_ids=title[modalities[:, 1] == 1],
                attention_mask=title_attn[modalities[:, 1] == 1],
            )[-1]
            title_feats[modalities[:, 1] == 1] = self.title_layernorm(
                self.title_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )

        image_text_feats = third_party.zeros_like(comment_feats)
        if (
            image_text is not None
            and (modalities[:, 2] == 1).sum().item()
            and image_text_attn.sum() > 2
        ):
            transformer_output = self.transformer(
                input_ids=image_text[modalities[:, 2] == 1],
                attention_mask=image_text_attn[modalities[:, 2] == 1],
            )[-1]
            image_text_feats[modalities[:, 2] == 1] = self.image_text_layernorm(
                self.image_text_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )
        image_feats = third_party.zeros_like(comment_feats)
        if image is not None and (modalities[:, 2] == 1).sum().item():
            image_feats[modalities[:, 2] == 1] = self.image_layernorm(
                self.image_proj(
                    self.feature_dropout(
                        self.image_cnn(image[modalities[:, 2] == 1]).reshape(
                            (modalities[:, 2] == 1).sum(), -1
                        )
                    )
                )
            )

        if "transformer_output" in locals():
            del transformer_output

        combine = third_party.cat(
            (comment_feats, title_feats, image_text_feats, image_feats), dim=1
        )

        projection = self.projection(combine)
        output = third_party.empty(
            len(projection), max(self.output_dims), device=projection.device
        )
        for task, dim in enumerate(self.output_dims):
            if dim == 1:
                output[task_indices[:, 0] == task, :1] = self.sigmoid(
                    self.task_outputs[task](projection[task_indices[:, 0] == task])
                )
            else:
                output[task_indices[:, 0] == task, :dim] = self.task_outputs[task](
                    projection[task_indices[:, 0] == task]
                )
        return output

    def save_model(self, dir):
        self.transformer.save_pretrained(dir)
        third_party.save(self.image_cnn.state_dict(), f"{dir}/image_cnn.pth")
        third_party.save(self.comment_proj.state_dict(), f"{dir}/comment_proj.pth")
        third_party.save(self.title_proj.state_dict(), f"{dir}/title_proj.pth")
        third_party.save(self.image_text_proj.state_dict(), f"{dir}/image_text_proj.pth")
        third_party.save(self.image_proj.state_dict(), f"{dir}/image_proj.pth")
        third_party.save(self.projection.state_dict(), f"{dir}/projection.pth")
        third_party.save(self.task_outputs.state_dict(), f"{dir}/task_output.pth")
        third_party.save(self.comment_layernorm.state_dict(), f"{dir}/comment_layernorm.pth")
        third_party.save(self.title_layernorm.state_dict(), f"{dir}/title_layernorm.pth")
        third_party.save(
            self.image_text_layernorm.state_dict(), f"{dir}/image_text_layernorm.pth"
        )
        third_party.save(self.image_layernorm.state_dict(), f"{dir}/image_layernorm.pth")

    def load_model(self, dir):
        self.transformer.from_pretrained(dir)
        self.image_cnn.load_state_dict(third_party.load(f"{dir}/image_cnn.pth"))
        self.comment_proj.load_state_dict(third_party.load(f"{dir}/comment_proj.pth"))
        self.title_proj.load_state_dict(third_party.load(f"{dir}/title_proj.pth"))
        self.image_text_proj.load_state_dict(third_party.load(f"{dir}/image_text_proj.pth"))
        self.image_proj.load_state_dict(third_party.load(f"{dir}/image_proj.pth"))
        self.projection.load_state_dict(third_party.load(f"{dir}/projection.pth"))
        self.task_outputs.load_state_dict(third_party.load(f"{dir}/task_output.pth"))
        import os

        # check to see if the model was previously saved with layernorm for compatability reasons
        if os.path.exists(f"{dir}/comment_layernorm.pth"):
            self.comment_layernorm.load_state_dict(
                third_party.load(f"{dir}/comment_layernorm.pth")
            )
            self.title_layernorm.load_state_dict(
                third_party.load(f"{dir}/title_layernorm.pth")
            )
            self.image_text_layernorm.load_state_dict(
                third_party.load(f"{dir}/image_text_layernorm.pth")
            )
            self.image_layernorm.load_state_dict(
                third_party.load(f"{dir}/image_layernorm.pth")
            )
