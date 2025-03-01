<div align="center">    

# Efficient Event-Based Semantic Segmentation via Exploiting Frame-Event Fusion: A Hybrid Neural Network Approach

</div>

## 环境配置
首先安装所需依赖:
```bash
pip install -r requirements.txt
```

## 数据准备
下载数据集 

[DDD17-Seg数据集]()
[DSEC-Semantic数据集]()
[M3ED-Semantic数据集]()

## 测试
权重文件

[DDD17-Seg数据集]()
[DSEC-Semantic数据集]()
[M3ED-Semantic数据集]()

```bash
python test.py --config-file local_configs/ddd17.yaml MODEL.WEIGHTS ./weights/ddd17_seg.pth
```
