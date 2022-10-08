#!/bin/bash
set -x

DATAHOME=${HOME}/cs/amr_qg/data
EXEHOME=${HOME}/cs/amr_qg/src/preprocess

#cd ${EXEHOME}
#python preprocess_raw_data.py ${DATAHOME}/dataset/hotpot_train_v1.1.json \
#                              ${DATAHOME}/dataset/hotpot_dev_distractor_v1.json \
#                              ${DATAHOME}/split_data

#得到AMR图
cd ${EXEHOME}/generate_amr_graph
#python get_merge_data.py ${DATAHOME}/split_data/data.train.add.json \
#                         ${DATAHOME}/amr_data/train_amr_add.json

#python get_merge_data.py ${DATAHOME}/split_data/data.valid.json \
#                         ${DATAHOME}/amr_data/valid_amr.json

#将amr图提取出来以后续简化
#python extract_amr.py    ${DATAHOME}/amr_data/train_amr.json \
#                         ${DATAHOME}/amr_data/train.amr

#python extract_amr.py    ${DATAHOME}/amr_data/valid_amr.json \
#                         ${DATAHOME}/amr_data/valid.amr

#简化amr图
#cd ${EXEHOME}/generate_amr_graph/NeuralAmr_NoAnon
#chmod a+x ./anonDeAnon_java.sh
#sed -i -e 's/\r$//' ./anonDeAnon_java.sh
#./anonDeAnon_java.sh anonymizeAmrFull true ${DATAHOME}/amr_data/train.amr
#./anonDeAnon_java.sh anonymizeAmrFull true ${DATAHOME}/amr_data/valid.amr  

#将简化后的图与原数据集合并
cd ${EXEHOME}/generate_amr_graph
python merge.py ${DATAHOME}/amr_data/train_amr.json \
                ${DATAHOME}/amr_data/train.amr.anonymized \
                ${DATAHOME}/merge_data/train_with_simple_amr.json

python merge.py ${DATAHOME}/amr_data/valid_amr.json \
                ${DATAHOME}/amr_data/valid.amr.anonymized \
                ${DATAHOME}/amr_data/valid_with_simple_amr.json