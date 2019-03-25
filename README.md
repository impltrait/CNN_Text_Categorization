### 我这也是借鉴别的写的，有啥不对的请联系我：

邮箱：admihao@163.com

![img](https://github.com/atucson/CNN_Text_Categorization/blob/master/doc/%E5%9E%83%E5%9C%BE%E9%82%AE%E7%AE%B1.png?raw=true)


-- -


![img](https://github.com/atucson/CNN_Text_Categorization/blob/master/doc/%E6%B5%81%E7%A8%8B%E8%AF%B4%E6%98%8E.jpg?raw=true)


-- -


[参考网址](https://zhuanlan.zhihu.com/p/35944222)

- text_cnn.py：网络结构设计

- train.py：网络训练

- eval.py：预测&评估

- data_helpers.py：数据预处理


训练：python train.py

测试： python eval.py --eval_train --checkpoint_dir="./runs/1546508380/checkpoints/
      
#### 这里1546508380需要修改, 根据自己的信息写


## 方法解释：



### 关于json，pickle，itsdangerous中的loads\dumps的对比分析
查看点击：[参考网址](https://blog.csdn.net/Odyssues_lee/article/details/80921195)
