# BigCodec
>**Abstract:**<br>
We present BigCodec, a low-bitrate neural speech codec. While recent neural speech codecs have shown impressive progress, their performance significantly deteriorates at low bitrates (around 1 kbps). Although a low bitrate inherently restricts performance, other factors, such as model capacity, also hinder further improvements. To address this problem, we scale up the model size to 159M parameters that is more than 10 times larger than popular codecs with about 10M parameters. Besides, we integrate sequential models into traditional convolutional architectures to better capture temporal dependency and adopt low-dimensional vector quantization to ensure a high code utilization. Comprehensive objective and subjective evaluations show that BigCodec, with a bitrate of 1.04 kbps, significantly outperforms several existing low-bitrate codecs. Furthermore, BigCodec achieves objective performance comparable to popular codecs operating at 4-6 times higher bitrates, and even delivers better subjective perceptual quality than the ground truth.

[[Paper]](https://arxiv.org/abs/2409.05377) [[Demo]](https://aria-k-alethia.github.io/bigcodec-demo/)

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

BigCodec extracts a single token to represent each frame of the utterance. Refer to `inference.py` to find how to get the code.

## Citation

## LICENCE
MIT
