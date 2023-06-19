### 使用说明

工具箱使用的python版本为3.7.6，包含三个主要文件：

``` python
ml_factory.py  # 工厂模块，定义模型工厂类，用于调用相关python库生成模型类，模型中一些需要调节的参数也在这里
run_ml.py  # 定义模型类和方法，目前包括模型fitting和validation
scoring_method.py  #定义一些评价函数，包括R方，RMSE，MAE等
```

使用方法：

**一、首先进入项目根目录下通过requirements.txt安装需要的库：**

1. pip3 install pipreqs 安装pipreqs库
2. pipreqs ./ --encoding=utf-8 或 pipreqs ./ --encoding=utf-8 --force
3. **pip3 install -r requirements.txt #将requirements.txt文件里的包全部安装**

**二、通过pandas读取数据**

```python
# 通过pandas读取数据，保证数据表格的最后一列为目标真值，其他列为输入特征值
import pandas as pd

# 读取csv
dataset = pd.read_csv('你的数据目录')
# 或读取excel
dataset = pd.read_excel('你的数据目录')
```

数据的结构类似下图，如使用神经网络需要先将数据归一化，集成学习模型不需要归一化。

![image-20210621103918892](https://gitee.com/Bearccd/blogimage/raw/master/img_byWindows/image-20210621103918892.png)

**三、在项目中引入工厂模块并调用合适的模型 **

目前主要具备模型拟合和评估功能：

``` python
import ml_factory

# model_name是模型名称，目前暂时支持RF，AdaBoost，GBDT，XGBoost，LightGBM，GRNN，LinearRegression
# 如需要仔细调参，目前还是得在ml_factory模块中自行修改
modal = ml_factory.ModalFactory(model_name)
modal_ins = modal.getModalClassInstance()
if modal_ins is None:
	print('No such modal')
	return
# 加载数据
modal_ins.load_data(dataset)
# fitting
[PM,train_estimated,modalR] = modal_ins.modal_fitting()
# 十折validation
[val,val_estimated,best_modal,best_train_target,best_train_estimated] = modal_ins.modal_tenFolder()


```

或参考

<font color="red">test_instant.py示例</font>

