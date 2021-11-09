#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : main.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 09.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>

from app import app

if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)
