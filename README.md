# 汽车检测项目实战
## 项目实现方式
1.利用项目提供的数据集，使用第7周的作业代码为模板，训练出一个checkpoint，并根据checkpoint生成相应的pb模型文件

2.利用根据步骤1的pb模型文件创建inference模型，使得其可以对任意汽车图片进行分类检测

3.利用ssd_mobilenet_v1_coco的预训练模型对图片进行目标检测，实现只针对汽车进行bondingbox圈出的功能，同时对圈出的部分再利用步骤2的模型进行分类检测并给与输出

具体实现成果见视频：vedio.avi 视频文件

## 项目实现过程详述：
### ①分类检测训练
训练结果详见：https://www.tinymind.com/executions/y3nbpngo

代码主要利用第7周的作业代码，根据本次数据集的实际情况对dataset和一些参数进行了修改。训练使用nceptionv4网络，0.001的学习率，利用google提供的预训练模型，经过20几个小时的训练后，达到eval/Accuracy[0.834472656]和eval/Recall_5[0.937744141]的效果。

训练完成后下载最终的checkpoint （训练输出结果中model.ckpt-46860的三个文件），为后续inference提供依据。

### ②目标检测实现
利用第8周object_detection的代码里object_detection_tutorial.ipynb为模板，采用ssd_mobilenet_v1_coco提供的frozen_inference_graph.pb文件为依据实现对任意图片仅把‘car’（汽车）用bondingbox圈出。这里使用了1个trick，就是修改了object_detection/visualization_utils.py中的代码只有在颜色为'Aquamarine'时，才实施画bondingbox的动作。（颜色'Aquamarine'在mobilenet中恰好为‘car’汽车分类的对应颜色）

### ③分类检测模型的导出
参考第7周课程视频，使用slim下export_inference_graph.py工具做成相应的inception_v4的不包含权重的pb模型文件。
再利用tensorflow下python/tools/freeze_graph.py工具，结合分类步骤①检测训练的checkpoint即model.ckpt-46860，冻结生成后续用于inferece的包含权重的模型pb文件frozen_46860.pb

1）使用export_inference_graph.py工具，参照给出的数据集导出对应的inceptionV4不带权重的模型，生成inception_v4_inf_graph.pb文件

python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v4 \
  --dataset_name=pj_vehicle \
  --batch_size=1 \
  --output_file=/tmp/inception_v4_inf_graph.pb


2）使用checkpoint对模型进行冻结，生成frozen_46860.pb文件

python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py \
  --input_graph=/tmp/inception_v4_inf_graph.pb \
  --input_binary=True \
  --datasetname=pj_vehicle \ 
  --input_checkpoint=/home/mengxi/Desktop/projectvechiledetection/ckpt/model.ckpt-46860  \ 
  --output_graph=/home/mengxi/Desktop/projectvechiledetection/ckpt/frozen_46860.pb \  
  --output_node_names=output 

### ④改写官方提供classify_image.py文件实现分类检测的inference应用

详细代码见代码里classify_image.py文件，完成改写后下载了一张test.jpg的汽车图片，在命令终端中运行

python3 classify_image.py \
--model_file=/home/mengxi/Desktop/projectvechiledetection/ckpt/frozen_46860.pb \
--label_file=labels.txt  \
--image_file=test.jpg

显示检测结果如下：

id:[185] name:[哈弗-H6] (score = 0.66801)

id:[183] name:[哈弗-H2] (score = 0.03234)

id:[100] name:[众泰-SR9] (score = 0.01575)

id:[619] name:[起亚-极睿] (score = 0.01003)

id:[248] name:[奔腾-X80] (score = 0.00798)

说明模型inference成功。


### ④整合步骤②中改写的object_detection_tutorial.ipynb和步骤③中的classify_image.py，做成detection and classify.ipynb，直接在jupyter notebook中进行使用。

进行单张图片目标检测和车型分类，只需要下载相应的图片，并把该图片的路径写到image_path中即可。（视频的做法是下载任意汽车图片并覆盖掉原来路径中的图片）


## 心得总结
通过本次项目，对于模型的训练，保存和导出以及运用有了比较深刻的理解。对于各个任务的整合也有了初步的认识。
本次由于时间有限，只是把目标检测和分类检测进行了一个简单的合并，但是结果的输入输出还不是很友好，看起来不是很方便。
如果能够把车型检测结果TOP1的结果能够直接在图片上显示出来恐怕效果会好很多，这需要对框架有更深刻的理解和更娴熟的技巧，个人还需要再接再厉。
另外，本次的结果演示是在虚拟机上演示的，使用的也是CPU，所以运行效率上不太尽如人意，这也是值得优化的地方。




# projectvechiledetection3
