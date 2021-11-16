# HCRN-BERT
This repo. includes a BERT-version [HCRN](https://arxiv.org/abs/2002.10698)(Le et. al., CVPR'20) implementation. Code is tested on [NExT-QA](https://github.com/doc-doc/NExT-QA) dataset. Official GloVe-version implementation is found [here](https://github.com/thaolmk54/hcrn-videoqa).
## Set-up
Please follow [official repo](https://github.com/thaolmk54/hcrn-videoqa) to set up the environment.

## Data Preparation
Please download the pre-computed video features for NExT-QA [here](https://drive.google.com/file/d/10vWHcfUXNiV4yqrgRl3sieAXHiRoE36i/view?usp=sharing). Then, extract the features into ```data/feats/HCRN/```. ```data/``` should be in the same directory as this repo.: ```workspace/HCRN-BERT```, ```workspace/data```.
Please follow [NExT-QA repo](https://github.com/doc-doc/NExT-QA) to download and prepare the finetuned BERT features.


## Usage
Once the data is ready, you can easily run the code. 
```
>python train.py
```
## License
Please follow the [official repo](https://github.com/thaolmk54/hcrn-videoqa). 
