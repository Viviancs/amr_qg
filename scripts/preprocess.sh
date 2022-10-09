set -x

DATAHOME=${HOME}/cs/qg/data
EXEHOME=${HOME}/cs/qg/src

cd ${EXEHOME}
python preprocess_data.py /data1/lkx/cs/amr_qg/data/merge_data/train_with_simple_amr.json \
                     /data1/lkx/cs/amr_qg/data/merge_data/valid_with_simple_amr.json \
                     ${DATAHOME}/train_data/preprcessed_sequence_data.pt \
                     ${DATAHOME}/train_data/preprcessed_graph_data.pt \
                     ${DATAHOME}/train_data/train_dataset.pt \
                     ${DATAHOME}/train_data/valid_dataset.pt