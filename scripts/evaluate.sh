set -x

DATAHOME=${HOME}/cs/amr_qg/data
EXEHOME=${HOME}/cs/amr_qg/src

cd ${EXEHOME}

#pip install git+https://github.com/salaniz/pycocoevalcap
python evaluate_metrics.py /data1/lkx/cs/amr_qg/logs/prediction.txt