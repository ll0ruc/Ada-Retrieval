# Ada-Retrieval

This is our PyTorch implementation for Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential
Recommendations (accepted in [AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/28712)). This project is developed based on [UniRec](https://github.com/microsoft/unirec)

# Requirements
Environments:
```
python==3.8
pytorch==1.12.1
cudatoolkit==11.3.1
```
Install environments by:
```
pip install -r requirements.txt
```

# Dataset
## Get our prepared dataset
We have placed the processed Beauty data in the /data/Beauty directory.

## Process dataset
> preprocess/

You can also use the pipeline in `preprocess/` to generate the processed dataset automatically. This pipeline includes:
- downloading the Beauty/Sports/Yelp dataset from [FMLP-Rec](https://github.com/Woeee/FMLP-Rec)
- modify the corresponding parameters about data in prepare_data.py.
- python prepare_data.py


# Quick Start

## train base model (e.g. SASRec)
You can use the shell command to train the model (you need to change `LOCAL_ROOT` to your path)
You need fit the `is_adaretrieval=0`
```
cd ./src/shell
bash main.sh
```
Then you will get a pre-trained model in ./output/base/Beauty/SASRec/checkpoint/SASRec-SASRec.pth

## finetune Ada-Retrieval

Train Ada-Ranker in the second-stage.
You need fit the `is_adaretrieval=1` and `load_best_model=1` to download your pre-trained model from the first stage
(notice that the model_file path should be consistent with the generated model in the first stage)

```
bash main.sh
```

See more details of main files in `./src/main/`.

# Output
Output path will be like this:
```
AdaRetrieval/output/
    - Ada-Retrieval/
        - Beauty/SASRec/
            - checkpoint/SASRec-SASRec.pth
            result_SASRec_timestamp.tsv
            SASRec_timestamp.txt
    - Base/
        - Beauty/SASRec/
            - checkpoint/SASRec-SASRec.pth
            result_SASRec_timestamp.tsv
            SASRec_timestamp.txt
```


This framework includes 5 basic sequential recommender models: GRU4Rec, SASRec, NextItNet, SRGNN, FMLPRec.

# Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```
@inproceedings{li2024ada,
  title={Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations},
  author={Li, Lei and Lian, Jianxun and Zhou, Xiao and Xie, Xing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={8},
  pages={8670--8678},
  year={2024}
}
```
