set -x

DATAHOME=${HOME}/cs/qg/data
EXEHOME=${HOME}/cs/qg/src

cd ${EXEHOME}
python preprocess_data.py /data1/lkx/cs/qg/data/train_data/mini/train_with_simple_amr.json \
                     /data1/lkx/cs/qg/data/train_data/mini/valid_with_simple_amr.json \
                     ${DATAHOME}/train_data/mini/preprcessed_sequence_data_mini.pt \
                     ${DATAHOME}/train_data/mini/preprcessed_graph_data_mini.pt \
                     ${DATAHOME}/train_data/mini/train_dataset_mini.pt \
                     ${DATAHOME}/train_data/mini/valid_dataset_mini.pt