# Multimodal Feature Evaluation Benchmark


This is a benchmark for multimodal feature evaluation, which is introduced in the paper ["RedFeat: Recoupling detection and description for multimodal feature learning"](https://arxiv.org/abs/2205.07439). The benchmark consists of three datasets, including VIS-NIR, VIS-IR and VIS-SAR.


## Subsets
The information of data collected in our benchmark is summerized in the table below, in which "channel" denotes the number of channels of visible\ the other type image, "number" denotes the number of image pairs in the train\test split.  
|         | Image type                        | Channel |  Number  |   Size  |          Characteristic         | Reference |
|---------|-----------------------------------|:-------:|:--------:|:-------:|:--------------------------:|-----------|
| VIS-NIR | Visible& Near infrared            |   3\1   |  345\128 | 983x686 | Multiple scenes            | [1]       |
| VIS-IR  | Visible& Long-wave infrared       |   3\1   |  211\47  | 533x321 | Road video at night        | [2,3]     |
| VIS-SAR | Visible& Synthetic aperture radar |   1\1   | 2011\424 | 512x512 | Remotely sensed by statilite | [4]       |

[1] Matthew Brown and Sabine Süsstrunk. Multi-spectral sift for scene category recognition. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit., pages 177–184, 2011.

[2] Cristhian A Aguilera, Angel D Sappa, and Ricardo Toledo. Lghd: A feature descriptor for matching across non-linear intensity variations. In Proc. IEEE Int. Conf. Image Process., pages 178–181, 2015.

[3] Han Xu, Jiayi Ma, Zhuliang Le, Junjun Jiang, and Xiaojie Guo. Fusiondn: A unified densely connected network for image fusion. In Proc. AAAI Conf. Artif. Intell., pages 12484–12491, 2020.

[4] Yuming Xiang, Rongshu Tao, Feng Wang, Hongjian You, and Bing Han. Automatic registration of optical and sar images via improved phase congruency model. IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens., 13:5847–5861, 2020.

## Build the Benchmark
Firstly, clone the repository to your workplace by 

```bash
git clone https://github.com/ACuOoOoO/Multimodal_Feature_Evaluation
```

Secondly, manually download datasets from links below:

(Nirscene) http://matthewalunbrown.com/nirscene/nirscene.html

(RoadScene) https://github.com/jiayi-ma/RoadScene

(LWIR\RGB) https://github.com/ngunsu/LGHD

(OSdataset) https://drive.google.com/file/d/1hgtj56HUlGSFbl5K1Sw16tKLx6rGXDbT/view?usp=sharing or
https://pan.baidu.com/s/17dbmuuVqgoYzkmw7Qg92tQ 提取码：bkro

Thirdly, clone the zips or the folders into our repository as
```
Multimodal Feature Evaluation                          
 └── VIS-NIR 
       ├── nirscene1.zip
       ...
 └── VIS-IR
       ├── RoadScene
       ├── lghd_icip2015_rgb_lwir.zip
       ...
 └── VIS-SAR
       ├── OSdataset.zip
       ...
```

Finally, run build.py in child folders as
```
python VIS_NIR/build.py
python VIS_IR/build.py
python VIS_SAR/build.py
```

