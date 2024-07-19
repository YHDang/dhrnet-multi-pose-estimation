# train on coco with 2 gpus
python tools/train.py --cfg experiments/coco.yaml --gpus 2,3
nohup python -u tools/train.py --cfg experiments/coco.yaml --gpus 2,3 >logs/3-coco-DHRNet-16-RandomSeed.log 2>&1 &
tail -f logs/3-coco-DHRNet-16-RandomSeed.log

# evaluate on coco val set with 2 gpus
# python tools/valid.py --cfg experiments/coco.yaml --gpus 2,3 TEST.MODEL_FILE model/coco/checkpoint.pth.tar

# evaluate on coco test-dev set with 2 gpus (submit to codalab)
# python tools/infer_coco_testdev.py --cfg experiments/coco.yaml --gpus 0,1 TEST.MODEL_FILE model/coco/checkpoint.pth.tar