# Bangla-Headline-Categorizer-App
Bangla Article Headline categorizer made with Streamlit using Transformer models. The pretrained model file is stored in a S3 bucket which is publicly accessible.
## Demo
  [![Alt text](https://github.com/MotaharMahtab/Bangla-Clickbait-Detector-App/blob/main/clickbait_detection_demo.gif)

[Streamlit website](https://www.streamlit.io/)

## Installation
It is recommended to use a virtual environment before installing the dependencies
```console
pip install -r requirements.txt
```
## Download Pretrained Model
Pretrained model is served as S3 bucket which will be made public when the corresponding thesis paper is published
```console
import wget
wget.download(s3_bucket_link)
```
Run
```console
streamlit run main.py
```
