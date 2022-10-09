set -x

DATAHOME=${HOME}/cs/qg/data
EXEHOME=${HOME}/cs/qg/src

cd ${EXEHOME}
python train.py \
       -sequence_data '/data1/lkx/cs/qg/data/train_data/mini/preprcessed_sequence_data_mini.pt' \
       -graph_data '/data1/lkx/cs/qg/data/train_data/mini/preprcessed_graph_data_mini.pt' \
       -train_dataset '/data1/lkx/cs/qg/data/train_data/mini/train_dataset_mini.pt' \
       -valid_dataset '/data1/lkx/cs/qg/data/train_data/mini/valid_dataset_mini.pt' \
       -checkpoint "/data1/lkx/cs/qg/model/classifier_cls_93.99113_accuracy.chkpt" \
       -epoch 1 \
       -batch_size 4 -eval_batch_size 4 \
       -training_mode generate \
       -max_token_src_len 200 -max_token_tgt_len 50 \
       -sparse 0 \
       -copy \
       -coverage -coverage_weight 0.4 \
       -node_feature \
       -feature \
       -d_word_vec 256 \
       -d_seq_enc_model 256 -d_graph_enc_model 256 -n_graph_enc_layer 3 \
       -d_k 32 -brnn -enc_rnn gru \
       -d_dec_model 256 -n_dec_layer 1 -dec_rnn gru \
       -maxout_pool_size 2 -n_warmup_steps 10000 \
       -dropout 0.5 -attn_dropout 0.1 \
       -gpus 0 \
       -save_mode best -save_model "/data1/lkx/cs/qg/model/generator_demo" \
       -log_home '/data1/lkx/cs/qg/logs' \
       -logfile_train "/data1/lkx/cs/qg/logs/train_generator" \
       -logfile_dev "/data1/lkx/cs/qg/logs/valid_generator" \
       -translate_ppl 15 \
       -curriculum 0  -extra_shuffle -optim adam \
       -learning_rate 0.00025 -learning_rate_decay 0.75 \
       -valid_steps 100 \
       -decay_steps 500 -start_decay_steps 5000 -decay_bad_cnt 5 \
       -max_grad_norm 5 -max_weight_value 32 \
       -translate_steps 2