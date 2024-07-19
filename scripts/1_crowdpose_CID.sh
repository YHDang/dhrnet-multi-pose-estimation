
# train on crowdpose with 2 gpus
# python tools/train.py --cfg experiments/crowdpose.yaml --gpus 2,3

# evaluate on crowdpose test set with 2 gpus
python tools/valid.py --cfg experiments/crowdpose.yaml --gpus 2,3 TEST.MODEL_FILE runs/crowdpose/DHRNet-no-random-seed/model_best.pth.tar


nohup python -u tools/train.py --cfg experiments/crowdpose.yaml --gpus 2,3 >logs/1-crowdpose-DHRNet-Pretrain.log 2>&1 &
tail -f logs/1-crowdpose-DHRNet-Pretrain.log