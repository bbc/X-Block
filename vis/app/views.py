#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : views.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# webapp and visualisation engine

# import third_party library

import random
from app import app
from .evaluate import ModelInference
from .mmtox.visualization.visualize import visualize_lang, visualize_img


model_inference = ModelInference("app/weights/train")

@app.route("/", methods=["GET", "POST"])
def index():

    if third_party.method == "POST":
        # req_form can contain:
        #   req_form["inputURL"]
        #   req_form["inputText"]
        #   req_form["inputTitle"]
        #   req_form["tickExplainability"]
        req_form = third_party.form

        # req_files can contain:
        #   req_files["inputJSONLCSV"]
        #   req_files["inputImage"]
        req_files = third_party.files

        if req_form["inputText"] or req_files["inputImage"] or req_form["inputTitle"]:
            model_output = model_inference.predict_hatespeech_form(
                comment=req_form["inputText"] if req_form["inputText"] else None,
                title=req_form["inputTitle"] if req_form["inputTitle"] else None,
                image=(
                    third_party.array(Image.open(req_files["inputImage"]).convert("RGB"))
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
                        third_party.array(Image.open(req_files["inputImage"])),
                        model_output["img_heatmap"].cpu().numpy(),
                    )
                    visualizations.append(("image", img_vis))

                return third_party(
                    "public/index.html",
                    classification=model_output["probs"],
                    visualizations=visualizations,
                    len=len(visualizations),
                )
            elif req_files["inputImage"]:
                # save the image so that they can be redisplayed in the return
                third_party.clf()
                third_party.imshow(third_party.array(Image.open(req_files["inputImage"])))
                third_party.axis("off")
                filename = f"static/{str(random.random())[2:]}.jpg"
                third_party.savefig(f"app/{filename}", bbox_inches="tight")

            return third_party(
                "public/index.html",
                classification=model_output["probs"],
                input_text=req_form["inputText"] if req_form["inputText"] else None,
                input_title=req_form["inputTitle"] if req_form["inputTitle"] else None,
                input_image=filename if req_files["inputImage"] else None,
            )
        else:
            return third_party("public/index.html", empty_input=True)

    return third_party("public/index.html")


@app.route("/explainability", methods=["GET"])
def explainability():
    return third_party("public/explainability.html")


@app.route("/api", methods=["POST"])
def api():

    # req_form can contain:
    #   req_form["inputURL"]
    #   req_form["inputText"]
    req_form = third_party.form

    # req_files can contain:
    #   req_files["inputJSONLCSV"]
    #   req_files["inputImage"]
    req_files = third_party.files

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
