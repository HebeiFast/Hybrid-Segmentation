<div align="center">    

# Efficient Event-Based Semantic Segmentation via Exploiting Frame-Event Fusion: A Hybrid Neural Network Approach

</div>

## Environment Setting
```bash
pip install -r requirements.txt
```

## Dataset Preparation

[DDD17-Seg数据集]()
[DSEC-Semantic数据集]()
[M3ED-Semantic数据集]()

## Evaluation
Weight Files

[DDD17-Seg数据集]()
[DSEC-Semantic数据集]()
[M3ED-Semantic数据集]()

```bash
python test.py --config-file local_configs/ddd17.yaml MODEL.WEIGHTS ./weights/ddd17_seg.pth
```
