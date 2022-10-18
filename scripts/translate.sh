DATAHOME=${HOME}/cs/qg/data
EXEHOME=${HOME}/cs/qg/src

cd ${EXEHOME}
python translate.py \
       -model "/data1/lkx/cs/qg/model/generator/generator_grt_9.2959_bleu4.chkpt" \
       -sequence_data '/data1/lkx/cs/qg/data/train_data/preprcessed_sequence_data.pt' \
       -graph_data '/data1/lkx/cs/qg/data/train_data/preprcessed_graph_data.pt' \
       -valid_data '/data1/lkx/cs/qg/data/train_data/valid_dataset.pt' \
       -output "/data1/lkx/cs/qg/logs/prediction.txt" \
       -beam_size 5 \
       -batch_size 8 \
       -gpus 2