# PDFparser
Here is a demo for PDF parser (Including OCR, object detection tools). 
PDF module recognition, extraction of multi-level headings, and more.
## Requirements
Firstly, I strongly recommend testing it on Linux.
```
pip install -r requirements
pip install "unstructured[pdf]"

sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

# using layoutparser tool and download the CV models (Detectron2)
pip install layoutparser torchvision && pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"

# layoutparser also supports paddle tool 
pip install "layoutparser[paddledetection]"
```
For unstructured installation, please refer to [here](https://unstructured-io.github.io/unstructured/installation/full_installation.html).
More details in [layoutparser](https://github.com/Layout-Parser/layout-parser/blob/main/installation.md).


## How to use
```python
# Extraction of Multi-level Headings
python multi_title.py

# Extraction other things
python parser.py
```
## Visualization of Extracted Multi-level Headings
![multi-level headings](multi_title_demo.png)

## Notes
Here is a detailed Chinese [blog](https://zhuanlan.zhihu.com/p/652975673). 
I apologize; due to project constraints, I can only share a portion of the code. However, feel free to ask any questions.

## Reference
- https://zhuanlan.zhihu.com/p/652975673
- https://unstructured.io/
- https://github.com/Layout-Parser/layout-parser/tree/main
- https://github.com/PaddlePaddle/PaddleDetection
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/tesseract-ocr/tesseract
