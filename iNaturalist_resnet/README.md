## iNaturalist Challenge(2018) with resnet

## Introduction
This release focuses on AI based graphic classification. We train resnet(152/101/50 layers) for iNaturalist Challenge at FGVC 2018 with [tensorpack](https://github.com/ppwwyyxx/tensorpack#toc0), which is a training interface based on TensorFlow.

## Result
On inaturalist-2018 Dataset, we train resnet(50/101/152) respectively，the result is as follows:

|Model Name|<sub>train-error-top1</sub>|<sub>train-error-top3</sub>|<sub>val-error-top1</sub>|<sub>val-error-top3</sub>|
|--------------------|---------|----------|---------|---------|
|<sub>Resnet50 </sub>| 0.13361 | 0.061188 | 0.399   | 0.24171 |
|<sub>Resnet101</sub>| 0.105   | 0.061306 | 0.37014 | 0.21371 |
|<sub>Resnet152</sub>| 0.11464 | 0.059394 | 0.35454 | 0.20024 |

## Installation, Data Preparation, Training and Testing

## Disclaimer
* The code is tested on a server with `188.00` GB memory, and `40` core cpu. Data storages in SSD.
* we train the model with 8 Pascal Titian XP gpu, for resnet50 the batch is `32*8=256`, for resnet101/152 the batch is `24*8=192`

## Install
Dependencies:
+ python3. We recommend using Anaconda as it already includes many common packages.
+ Python bindings for OpenCV (Optional, but required by a lot of features)
+ TensorFlow >= 1.3.0 (Optional if you only want to use `tensorpack.dataflow` alone as a data processing library)
```
# install git, then:
pip install -U git+https://github.com/ppwwyyxx/tensorpack.git
# or add `--user` to avoid system-wide installation.
```

## Prepare data
data should be organized as follows(set path in config.py):
```
$HOME/DataSet/iNaturalist2018/
    |->ground_truth
    |    |train2018.json
    |    |val2018.json
    |    |test2018.json
    |->train_val2018
    	 |...
    |->test2018
         |...
```

## Train and Eval
* train script example:
```
python iNaturalist-resnet.py --data /home/huzhikun/DataSet/iNaturalist2018/ --batch 192 --mode resnet --gpu 0,1,2,3,4,5,6,7 -d 152
```
* eval script example:
```
python iNaturalist-resnet.py --eval --data /home/huzhikun/DataSet/iNaturalist2018/ --mode resnet --gpu 0,1,2,3 -d 152 --load train_log/iNaturalist-resnet-d152/model-205065
```

## Test with kaggle submit file(.csv)
* test script example:
```
python iNaturalist-resnet.py --test --data /home/huzhikun/DataSet/iNaturalist2018/ --mode resnet --gpu 7 -d 152 --load ./train_log/iNaturalist-resnet-d152/model-239190
```

## Citing
```
@misc{wu2016tensorpack,
  title={Tensorpack},
  author={Wu, Yuxin and others},
  howpublished={\url{https://github.com/tensorpack/}},
  year={2016}
}
```
