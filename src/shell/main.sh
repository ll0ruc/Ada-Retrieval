#!/bin/bash

LOCAL_ROOT="Yourlocalpath/Ada-Retrieval"

MY_DIR=$LOCAL_ROOT
ALL_DATA_ROOT="$LOCAL_ROOT/data"
OUTPUT_ROOT="$LOCAL_ROOT/output"
# default parameters for local run
MODEL_NAME='SASRec' # ["GRU4Rec", "SASRec", "NextItNet", "SRGNN", "FMLPRec"]
DATASET_NAME="Beauty" #["Beauty", "Sports", "Yelp"]
is_adaretrieval=0
reco_batch_cnt=5
lambda=0.3
verbose=2
learning_rate=0.001
epochs=200
weight_decay=0
dropout_prob=0.1777
n_sample_neg_train=100
max_seq_len=50
embedding_size=64
cd $MY_DIR
export PYTHONPATH=$PWD

if [ $is_adaretrieval -eq 0 ]; then
    OUTPUT_PATH=$OUTPUT_ROOT"/Base/"$DATASET_NAME/$MODEL_NAME
else
    OUTPUT_PATH=$OUTPUT_ROOT"/Ada-Retrieval/"$DATASET_NAME/$MODEL_NAME
    PRE_TRAIN_PATH=$OUTPUT_ROOT"/Base/"$DATASET_NAME/$MODEL_NAME
fi

mkdir -p $OUTPUT_PATH
### train ###################################
python src/main/main.py \
    --config_dir="src/config" \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME \
    --output_path=$OUTPUT_PATH \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --max_seq_len=$max_seq_len \
    --has_item_bias=1 \
    --epochs=$epochs  \
    --batch_size=1024 \
    --n_sample_neg_train=$n_sample_neg_train \
    --valid_protocol='one_vs_all' \
    --test_protocol='one_vs_all' \
    --grad_clip_value=10 \
    --weight_decay=$weight_decay \
    --user_history_filename="user_history" \
    --user_history_file_format="user-item_seq"  \
    --shuffle_train=1 \
    --seed=366849 \
    --early_stop=10 \
    --embedding_size=$embedding_size \
    --hidden_size=$embedding_size \
    --num_workers=4 \
    --num_workers_test=0 \
    --verbose=$verbose \
    --hidden_dropout_prob=$dropout_prob \
    --attn_dropout_prob=$dropout_prob \
    --n_layers=2 \
    --n_heads=4 \
    --is_adaretrieval=$is_adaretrieval \
    --metrics="['hit@50;100', 'ndcg@50;100']" \
    --key_metric="hit@50" \
    --reco_total_size=100 \
    --reco_batch_cnt=$reco_batch_cnt \
    --lambda=$lambda  \
    --checkpoint_dir=$OUTPUT_PATH"/checkpoint" \
    --load_best_model=0 \
    --model_file=$PRE_TRAIN_PATH"/checkpoint/SASRec-SASRec.pth" \
    --gpu_id=0
