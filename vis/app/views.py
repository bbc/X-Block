#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : views.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 09.11.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
#
# Copyright (c) 2020, Imperial College, London
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#   3. Neither the name of Imperial College nor the names of its contributors may
#      be used to endorse or promote products derived from this software without
#      specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# webapp and visualisation engine

from flask import Response, render_template, request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import random
from app import app
from .evaluate import ModelInference
from .xblock.visualization.visualize import visualize_lang, visualize_img


model_inference = ModelInference("app/weights/train")

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        # req_form can contain:
        #   req_form["inputURL"]
        #   req_form["inputText"]
        #   req_form["inputTitle"]
        #   req_form["tickExplainability"]
        req_form = request.form

        # req_files can contain:
        #   req_files["inputJSONLCSV"]
        #   req_files["inputImage"]
        req_files = request.files

        if req_form["inputText"] or req_files["inputImage"] or req_form["inputTitle"]:
            model_output = model_inference.predict_hatespeech_form(
                comment=req_form["inputText"] if req_form["inputText"] else None,
                title=req_form["inputTitle"] if req_form["inputTitle"] else None,
                image=(
                    np.array(Image.open(req_files["inputImage"]).convert("RGB"))
                    if req_files["inputImage"]
                    else None
                ),
                explainability=(req_form.get("tickExplainability") == "on"),
            )
            if req_form.get("tickExplainability") == "on":
                # create list of visualisations
                visualizations = []
                if req_form["inputTitle"]:
                    txt_vis = visualize_lang(
                        model_inference.tokenizer.tokenize(req_form["inputTitle"]),
                        model_output["title_heatmap"].cpu().numpy(),
                    )
                    visualizations.append(("title", txt_vis))
                if req_form["inputText"]:
                    txt_vis = visualize_lang(
                        model_inference.tokenizer.tokenize(req_form["inputText"]),
                        model_output["comment_heatmap"].cpu().numpy(),
                    )
                    visualizations.append(("comment", txt_vis))
                if req_files["inputImage"]:
                    img_vis = visualize_img(
                        np.array(Image.open(req_files["inputImage"])),
                        model_output["img_heatmap"].cpu().numpy(),
                    )
                    visualizations.append(("image", img_vis))

                return render_template(
                    "public/index.html",
                    classification=model_output["probs"],
                    visualizations=visualizations,
                    len=len(visualizations),
                )
            elif req_files["inputImage"]:
                # save the image so that they can be redisplayed in the return
                plt.clf()
                plt.imshow(np.array(Image.open(req_files["inputImage"])))
                plt.axis("off")
                filename = f"static/{str(random.random())[2:]}.jpg"
                plt.savefig(f"app/{filename}", bbox_inches="tight")

            return render_template(
                "public/index.html",
                classification=model_output["probs"],
                input_text=req_form["inputText"] if req_form["inputText"] else None,
                input_title=req_form["inputTitle"] if req_form["inputTitle"] else None,
                input_image=filename if req_files["inputImage"] else None,
            )
        else:
            return render_template("public/index.html", empty_input=True)

    return render_template("public/index.html")


@app.route("/explainability", methods=["GET"])
def explainability():
    return render_template("public/explainability.html")


@app.route("/api", methods=["POST"])
def api():

    # req_form can contain:
    #   req_form["inputURL"]
    #   req_form["inputText"]
    req_form = request.form

    # req_files can contain:
    #   req_files["inputJSONLCSV"]
    #   req_files["inputImage"]
    req_files = request.files

    if req_form["inputText"] and req_form["inputImage"]:
        hatespeech_classification = model_inference.predict_hatespeech_form(
            req_form["inputText"], req_files["inputImage"]
        )
    elif req_files["inputJSONLCSV"]:
        hatespeech_classification = model_inference.predict_hatespeech_file(
            req_form["inputJSONLCSV"]
        )
    else:
        # for scraping from a URL
        raise NotImplementedError

    return hatespeech_classification
