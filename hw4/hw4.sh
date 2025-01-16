#!/bin/bash

# TODO - run your inference Python3 code
python3 gaussian-splatting/render_inf.py -m 'inference_model' -s $1 --output_path $2