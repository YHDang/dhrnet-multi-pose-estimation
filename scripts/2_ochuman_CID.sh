
# train on crowdpose with 2 gpus
python tools/train.py --cfg experiments/ochuman_val.yaml --gpus 2,3

# evaluate on crowdpose test set with 2 gpus
python tools/valid.py --cfg experiments/ochuman_val.yaml --gpus 2,3 TEST.MODEL_FILE runs/ochuman/DHRNet-SE/model_best.pth.tar
python tools/valid.py --cfg experiments/ochuman_val.yaml --gpus 3 TEST.MODEL_FILE runs/ochuman/DHRNet-SE/model_best.pth.tar

CUDA_VISIBLE=2,3 nohup python -u tools/train.py --cfg experiments/ochuman_val.yaml --gpus 2,3 >logs/2-OCHuman-DHRNet-CBAM.log 2>&1 &
tail -f logs/2-OCHuman-DHRNet-CBAM.log

nohup python -u tools/train.py --cfg experiments/ochuman_val.yaml --is_train True --gpus 2,3 >logs/2-OCHuman-DHRNet-IJ.log 2>&1 &
tail -f logs/2-OCHuman-DHRNet-IJ.log