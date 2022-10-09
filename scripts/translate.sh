DATAHOME=${HOME}/cs/qg/data
EXEHOME=${HOME}/cs/qg/src

cd ${EXEHOME}
python translate.py \
       -model "/data1/lkx/cs/qg/model/generator_demo.chkpt" \
       -sequence_data '/data1/lkx/cs/qg/data/train_data/mini/preprcessed_sequence_data_mini.pt' \
       -graph_data '/data1/lkx/cs/qg/data/train_data/mini/preprcessed_graph_data_mini.pt' \
       -valid_data '/data1/lkx/cs/qg/data/train_data/mini/valid_dataset_mini.pt' \
       -output "/data1/lkx/cs/qg/logs/prediction.txt" \
       -beam_size 5 \
       -batch_size 8 \
       -gpus 0