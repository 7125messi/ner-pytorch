# ChineseNER
本项目使用
+ python3.6
+ pytorch 1.1.0

命名实体识别BiLSTM+CRF模型。

## 数据
data文件夹中有三个开源数据集可供使用，**玻森数据 (https://bosonnlp.com) 、1998年人民日报标注数据、MSRA微软亚洲研究院开源数据**。

其中**boson数据集有6种实体类型，人民日报语料和MSRA一般只提取人名、地名、组织名三种实体类型**。

先运行数据中的python文件处理数据，供模型使用。

## pytorch版
直接用的<a href="https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html">pytorch tutorial</a>里的Bilstm+crf模型.

运行train.py训练即可。由于使用的是cpu，而且也没有使用batch，所以训练速度超级慢。想简单跑一下代码的话，建议只使用部分数据跑一下。


## 准确率
boson数据集的f值在70%~75%左右，人民日报和MSRA数据集的f值在85%~90%左右。（毕竟boson有6种实体类型，另外两个只有3种）

* 开始训练

* 使用预训练的词向量

* 测试训练好的模型

* 文件级别实体抽取
使用 `python train.py input_file output_file` 进行文件级实体抽取。可以自动读取model文件夹中最新的模型，将`input_file`中的实体抽取出来写入`output_file`中。先是原句，然后是实体类型及实体（可按照需要修改）。