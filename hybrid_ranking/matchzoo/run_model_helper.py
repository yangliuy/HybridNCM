import os
import sys

if len(sys.argv) < 4:
    print 'please input params: phase_name config_name log_file_name GPU_CARD_NUM'
    exit(1)
phase_name = sys.argv[1]
config_name = sys.argv[2]
log_file_name = sys.argv[3]
GPU_CARD_NUM = sys.argv[4]

cmd = 'CUDA_VISIBLE_DEVICES=' + str(GPU_CARD_NUM ) + ' nohup python main_hybrid_ncm.py ' \
      '--phase ' + phase_name + ' --model_file ' + config_name + ' 2>&1 ' \
      '| tee ../../log/' + log_file_name
os.system(cmd)