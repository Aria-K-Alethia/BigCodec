# BigCodec

>**Abstract:**<br>

## Setup
Code is tested on `python3.9`

Please follow the following steps to setup your environment
1. Clone this repo
2. `pip install -r requirements.txt`
3. Download the pretrained checkpoint by
```bash
wget https://huggingface.co/Alethia/BigCodec/resolve/main/bigcodec.pt
```

## Inference
```bash
python inference.py --input-dir /input/wav/dir --output-dir /output/wav/dir --ckpt /ckpt/path
```
The above cmd reconstruct all `.wav` files under the input directory and write the results to the output directory using the checkpoint.

## Citation

## LICENCE
MIT
