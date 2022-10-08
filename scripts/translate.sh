DATAHOME=${HOME}/cs/amr_qg/data
EXEHOME=${HOME}/cs/amr_qg/src

cd ${EXEHOME}
python translate.py \
       -model "/data1/lkx/cs/amr_qg/model/generator_demo.chkpt" \
       -sequence_data '/data1/lkx/cs/amr_qg/data/train_data/test/preprcessed_sequence_data_mini.pt' \
       -graph_data '/data1/lkx/cs/amr_qg/data/train_data/test/preprcessed_graph_data_mini.pt' \
       -valid_data '/data1/lkx/cs/amr_qg/data/train_data/test/valid_dataset_mini.pt' \
       -output "/data1/lkx/cs/amr_qg/logs/prediction.txt" \
       -beam_size 5 \
       -batch_size 8 \
       -gpus 0