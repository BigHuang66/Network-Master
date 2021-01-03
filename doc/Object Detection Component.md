#   Object Detection Component

## Input

### Image

### Patches

### Image pyramid

## Backbones

### VGG16

- 2+2+2+3+3

### RestNet-50

- 3+4+6+3

### SpineNet

### EfficientNet-B0/B7

- AutoDL确定超参数

### CSPResNeXt50

### CSPDarknet53

### ...

## Neck

### Additional blocks

- SPP

  - Spatial Pyramid Pooling  空间金字塔池化
  - 源于RCNN中，原来本质是通过不同大小的kernel完成池化操作，并将结果concatenate解决不同尺寸特征图不匹配全连接层的问题，不同大小的池化解决了多尺度物体问题
  - yolo中上述不一定适用，更多的是下图做法
  - ![1603973913318](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603973913318.png)
  - 上述操作能够极大地增加感受野，分离出上下文特征

- ASPP

  ​         Atrous Spatial Pyramid Pooling 空洞空间金字塔池化

  ​		 论文链接：

  ​		 源于deeplab v2，kernel尺寸不变的情况下，空洞卷积能够进一步增大感受野，更多的融合上下文信息concatenate，实现多尺度信息，同时设置stride=1，kernel=3，padding=dilate，保证输入输出尺寸不变

  ![1603975067179](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603975067179.png)

- RFB

   还是和上述类似，主要改进 于**Inception** 结构，在每个子结构中添加空洞卷积，增大感受野，融合不同特征

  ![1603977686158](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603977686158.png)



###  Attention mechanism 

- SAM

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978262176.png" alt="1603978262176" style="zoom: 67%;" />

- modified SAM

  

   <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978277992.png" alt="1603978277992" style="zoom: 67%;" /> 

  

### Path-aggregation blocks

- FPN

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978394940.png" alt="1603978394940" style="zoom: 50%;" />

- PAN

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978409037.png" alt="1603978409037" style="zoom: 50%;" />

- NAS-FPN

   使用神经结构搜索找到一个不规则的特征网络拓扑，然后重复应用相同的块 

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978447142.png" alt="1603978447142" style="zoom: 50%;" />

- Fully-connected FPN

  ![1603978646812](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978646812.png)

- BiFPN

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603978541578.png" alt="1603978541578" style="zoom: 50%;" />

- ASFF

  原来的FPN add方式变成了add基础上多了一个可学习系数，该系数是自动学习的，可以实现自适应融合效果，类似于全连接参数。学习了在空间上过滤冲突信息以抑制梯度反传的时候不一致的方法，从而改善了特征的比例不变性。

   <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603979138140.png" alt="1603979138140" style="zoom: 67%;" />

- SFAM

   SFAM旨在聚合TUMs产生的多级多尺度特征，以构造一个**多级特征金字塔**。  在first stage，SFAM沿着    channel维度将拥有相同scale的feature map进行拼接，这样得到的每个scale的特征都包含了多个level的信息。然后在second stage，**借鉴SENet的思想，加入channel-wise attention**，以更好地捕捉有用的特征。 

  ![img](https://pic3.zhimg.com/80/v2-5862de6e7d2c50dd09b4a62e9b012036_720w.jpg)

  

## Heads

### Dense Prediction (one-stage)

- anchor based

	- RPN
	- SSD
	- YOLO
	- RetinaNet

- anchor free

	- CornerNet
	- CenterNet
	- MatrixNet
	- FCOS

### Sparse Prediction (two-stage)

- anchor based

	- Faster R-CNN
	- R-FCN
	- Mask R-CNN

- anchor free

	- RepPoints

## Bag of specials

### plugin modules and post-processing methods that only increase cost by a small amount but can significantly improve the accuracy

### enhance receptive field

- SPP

   见上

- ASPP

   见上

- RFB

   见上

### attention module

![1603979779423](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603979779423.png)

- channel-wise attention

	  Squeeze-and-Excitation

- point-wise attention

	  Spatial Attention Module

### feature integration

- FPN

  见上

- SFAM

  见上

- ASFF

  见上

- BiFPN

  见上

### activation function

- Sigmoid、ReLU、tanh

  ![1603981308127](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981308127.png)

  ![1603981319560](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981319560.png)

  

  ![1603980768079](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603980768079.png)

  ![1603981569751](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981569751.png)

  ![1603981579984](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981579984.png)

- LReLU、PReLU

   LReLu相对与ReLu对模型影响相差不大。故作者提出了PReLu, 通过学习ai，让模型更加完美。 

  ![1603980896059](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603980896059.png)

  ![1603980908187](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603980908187.png)

  

- ReLU6、hard-Swish

  ​         ReLU6(x)=min(max(0,x),6)   Mobile V1 为了在移动端设备float16的低精度的时候，也能有很好的数值分辨率 

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981008008.png" alt="1603981008008" style="zoom: 50%;" />

  

- SELU

  ![1603981767447](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981767447.png)

- Swish、Mish

  ​       Swish(x) = x*sigmoid(βx) 

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981444165.png" alt="1603981444165" style="zoom:50%;" />

  ​         Mish(x)= x*tanh(ln(1+e^x)) 

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1603981500024.png" alt="1603981500024" style="zoom:50%;" />

### post-processing

 来源： https://zhuanlan.zhihu.com/p/151914931

- #### 分类优先： NMS、Soft-NMS   (ICCV 2017) 

  Non-Maximum Suppression  非极大值抑制

   **传统NMS**有多个名称，据不完全统计可以被称为：Traditional / Original / Standard / Greedy NMS 

  可以理解为局部最大搜索，目的为去除冗余的检测框，保留最好的一个

  ![img](https://pic4.zhimg.com/80/v2-b327ffc98aa75448cde478137a1d2777_720w.jpg)

  **缺点**：

  1. 顺序处理的模式，计算IoU拖累了运算效率。

  2. 剔除机制太严格，依据NMS阈值暴力剔除。

  3. 阈值是经验选取的。

  4. 评判标准是IoU，即只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面。

     

     

  ​    **Soft-NMS** 针对上述第二点提出线性惩罚机制和高斯惩罚，最后再根据得分阈值保留检测框

  可能存在定位好而得分较低的框，即存在定位好坏与分类得分不一致的情况。

  ![[公式]](https://www.zhihu.com/equation?tex=s_i%3D%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Blc%7D+0%2C+%26+IoU%28M%2CB_i%29%5Cgeqslant+thresh%5C%5C+s_i%2C+%26+IoU%28M%2CB_i%29%3Cthresh%5C%5C+%5Cend%7Barray%7D%5Cright.)

  ![[公式]](https://www.zhihu.com/equation?tex=s_i%3D%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Blc%7D+s_i%281-IoU%28M%2CB_i%29%29%2C+%26+IoU%28M%2CB_i%29%5Cgeqslant+thresh%5C%5C+s_i%2C+%26+IoU%28M%2CB_i%29%3Cthresh%5C%5C+%5Cend%7Barray%7D%5Cright.)

  ![[公式]](https://www.zhihu.com/equation?tex=s_i%3Ds_ie%5E%7B-%5Cfrac%7BIoU%28M%2CB_i%29%5E2%7D%7B%5Csigma%7D%7D)

  **缺点**：

  1. 仍然是顺序处理的模式，运算效率比Traditional NMS更低。
  2. 对双阶段算法友好，而在一些单阶段算法上可能失效。
  3. 如果存在定位与得分不一致的情况，则可能导致定位好而得分低的框比定位差得分高的框惩罚更多(遮挡情况下)。
  4. 评判标准是IoU，即只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面。

- #### 定位优先：IoU-Guided NMS   (ECCV 2018) 

  ​       考虑定位好坏与分类置信度不一致的问题，引入IoU预测分支，将此结果作为NMS的筛选依据，每次挑选出最大定位置信度的框，并以此框进行IoU的剔除，自身得分和以此剔除的框最大置信度赋给该框。  

  **优点**：

  ​     IoU-Guided NMS有助于提高严格指标下的精度，如AP75, AP90。

  **缺点**：

  1. 顺序处理的模式，运算效率与Traditional NMS相同。
  2. 需要额外添加IoU预测分支，造成计算开销。
  3. 评判标准是IoU，即只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面。

- #### 加权平均：Weighted NMS  (ICME Workshop 2017) 

  传统 NMS每次迭代所选出的最大得分框未必是准确定位的（依旧是定位好坏与得分不一致）。与直接剔除不同，Weighted NMS是对坐标加权平均，加权平均的对象是本身与满足被剔除阈值的相邻框。

  ![[公式]](https://www.zhihu.com/equation?tex=M%3D%5Cfrac%7B%5Csum%5Climits_iw_iB_i%7D%7B%5Csum%5Climits_iw_i%7D%2C%5Cquad+B_i%5Cin%5C%7BB%7CIoU%28M%2CB%29%5Cgeqslant+thresh%5C%7D%5Ccup%5C%7BM%5C%7D)

  加权 权重为![[公式]](https://www.zhihu.com/equation?tex=w_i%3Ds_iIoU%28M%2CB_i%29)，表示得分与IoU的乘积

  **优点**：

  Weighted NMS通常能够获得更高的Precision和Recall，以本人的使用情况来看，只要NMS阈值选取得当，Weighted NMS均能稳定提高AP与AR，无论是AP50还是AP75，也不论所使用的检测模型是什么。

  **缺点**：

  1. 顺序处理模式，且运算效率比Traditional NMS更低。
  2. 加权因子是IoU与得分，前者只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面；而后者受到定位与得分不一致问题的限制。

- #### 方差加权平均：Softer-NMS （CVPR 2019)

  同样是加权平均思想，不同在于权重发生变化，以及引入box边界的不确定度。

  ![[公式]](https://www.zhihu.com/equation?tex=M%3D%5Cfrac%7B%5Csum%5Climits_iw_iB_i%2F%5Csigma_i%5E2%7D%7B%5Csum%5Climits_iw_i%2F%5Csigma_i%5E2%7D%2C%5Cquad+B_i%5Cin%5C%7BB%7CIoU%28M%2CB%29%5Cgeqslant+thresh%5C%7D%5Ccup%5C%7BM%5C%7D)

  其中权重![[公式]](https://www.zhihu.com/equation?tex=w_i%3De%5E%7B-%5Cfrac%7B%281-IoU%28M%2CB_i%29%29%5E2%7D%7B%5Csigma_t%7D%7D)不包含Si，而只与IoU有关

  **优点**：

  1. 可以与Traditional NMS或Soft-NMS结合使用。
  2. 通常可以稳定提升AP与AR。

  **缺点**：

  1. 顺序处理模式，且运算效率比Traditional NMS更低。
  2. 需要修改模型来预测方差。
  3. 加权因子是IoU与方差，前者依然只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面。

- #### 自适应阈值： Adaptive NMS（CVPR 2019）

  ​    当物体分布稀疏时，NMS大可选用小阈值以剔除更多冗余框；而在物体分布密集时，NMS选用大阈值，以获得更高的召回。 提出了密度预测模块，来学习一个框的密度 。

  一个GT框Bi的密度标签定义如下：![[公式]](https://www.zhihu.com/equation?tex=d_i%3A%3D%5Cmax%5Climits_%7BB_i%2CB_j%5Cin+GT%7DIoU%28B_i%2CB_j%29%2C+%5Cquad+i%5Cneq+j)
   模型的输出将变为 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%2Cw%2Ch%2Cs%2Cd%29) ，分别代表box坐标，宽高，分类得分，密度，其中密度 ![[公式]](https://www.zhihu.com/equation?tex=d) 越大，代表该框所处的位置的物体分布越密集，越有可能是遮挡严重的地方；反之密度 ![[公式]](https://www.zhihu.com/equation?tex=d) 越小，代表该框所处的位置的物体分布越稀疏，不太可能有遮挡。 

  论文以Traditionnal NMS和Soft-NMS的线性惩罚为基础，将每次迭代的NMS阈值更改如下：

  ![1604043293312](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604043293312.png)

  其中 thresh 代表最小的NMS阈值。

  **优点**：

  1. 可以与前面所述的各种NMS结合使用。
  2. 对遮挡案例更加友好。

  **缺点**：

  1. 与Soft-NMS结合使用，效果可能倒退 (受低分检测框的影响)。
  2. 顺序处理模式，运算效率低。
  3. 需要额外添加密度预测模块，造成计算开销。
  4. 评判标准是IoU，即只考虑两个框的重叠面积，这对描述box重叠关系或许不够全面。

- #### 中心点距离：DIoU-NMS （AAAI 2020）

  ![1604043374963](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604043374963.png)![1604043556739](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604043556739.png)

   第一种相比于第三种越不太可能是冗余框。基于该观点，研究者使用所提出的DIoU替代IoU作为NMS的评判准则，公式如下： 

  ![1604043407654](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604043407654.png)

  ![1604043442028](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604043442028.png)

   在实际操作中，研究者还引入了参数β用于控制 d2/c2 的惩罚幅度.

  1. 当 β->∞时，DIoU退化为IoU，此时的DIoU-NMS与Traditional NMS效果相当。
  2. 当β->0 时，此时几乎所有中心点不与M重合的框都被保留了。

  **优点**：

  1. 从几何直观的角度，将中心点考虑进来有助于缓解遮挡案例。
  2. 可以与前述NMS变体结合使用。
  3. 保持NMS阈值不变的情况下，必然能够获得更高recall (因为保留的框增多了)，至于precision就需要调整β来平衡了。
  4. 个人认为+中心点距离的后处理可以与DIoU/CIoU损失结合使用，这两个损失一方面优化IoU，一方面指引中心点的学习，而中心点距离学得越好，应该对这种后处理思想的执行越有利。

  **缺点**：

  1. 依然是顺序处理模式，运算效率低。
  2. DIoU的计算比IoU更复杂一些，这会降低运算效率。
  3. 在保持NMS阈值不变的情况下，使用DIoU-NMS会导致每次迭代剩余更多的框，这会增加迭代轮数，进一步降低运算效率。(经本人实测，DIoU-NMS是Traditional NMS 起码1.5倍耗时)

  #### 总结：

  1. 加权平均法通常能够稳定获得精度与召回的提升。
  2. 定位优先法，方差加权平均法与自适应阈值法需要修改模型，不够灵活。
  3. 中心点距离法可作为额外惩罚因子与其他NMS变体结合。
  4. 得分惩罚法会改变box的得分，打破了模型校准机制。
  5. 运算效率的低下可能会限制它们的实时应用性。

  

### Regularization

来源： https://www.zhihu.com/search?type=content&q=DropPath

- #### DropOut

  **完全随机drop** 忽略一部分的的特征检测器（节点设置为0），减少过拟合现象，减少特征检测器（隐层节点）间的相互作用

  在全连接层效果较明显，在卷积层不够明显，相当于”添加了噪声“，因为此时时候部分特征图内单元，由于卷积的特性，其信息可能包含在周围，无法达到用剩余单元特征 学习到分类的特征。

- #### DropConnect

  **随机 drop 连接，区别于随机 drop 神经元**

- #### DropPath

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604048239758.png" alt="1604048239758" style="zoom:67%;" />

- #### Cutout

  在下述数据增强部分也有描述。即**在 input 层 按照 spatial 块 随机 drop**

- ####  Stochastic Depth 

   **按res block随机扔 **，类似于模型剪枝

- #### Spatial DropOut

   随机drop一个 CHW 的特征 ，即 **随机drop调一个channel**![img](https://pic1.zhimg.com/v2-12d5fcbb475ab7bd969c490e3bf60474_b.jpg)

  

- #### DropBlock

  来源： https://zhuanlan.zhihu.com/p/54286076

   **每个feature map上按spatial块随机扔 **

   dropout 通常对全连接层很有效，对卷积层效果甚微。这可能是因为卷积层的激活单元在空间上是有相关性的，所以尽管采用了dropout,信息还是可以通过网络的其他地方获得。 

  ![img](https://pic3.zhimg.com/80/v2-792396f4c02168c5bf05db43a1181b32_720w.jpg)

   绿色区域主要蕴含了图像的语义信息，也就是狗所在的主要区域，通过图 b 的方式随机dropout效果并不好，因为相邻单元也包含了相关信息。按照图 C 的方式，移除整块区域，例如整个头或者脚，这样可以强制剩余单元学习到用于分类的特征。 

  和Dropout类似，DropBlock仅仅在训练时候使用，测试时不用。我们可以理解成测试时用的是训练时的子网络的集成网络，输出平均精度。每一个子网络都只能看到特征图的一部分，不能看到全部的连续特征。

  DropBlock有两个参数，**block_size,是我们要drop 的模块的大小**，当block_size=1时，就是常用的dropout。

  另一个参数  **γ 控制着要drop的特征数量**，假设keep_prob为我们想要保存特征的概率，则γ等于

  ![1604045906120](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604045906120.png)

   先按照二项分布采用，和dropout一样，采样出一些mask点，然后以这个点为中心，扩展成一个正方形区域，便成为block_size，为了保证每一个mask点都能扩展成一个正方形，本文仅仅在绿色区域中采样。 

  

### Normalization

### ![img](https://pic1.zhimg.com/80/v2-964f738bd81c0d41b451e258dae19c40_720w.jpg)

- BN

  用于神经网络激活层之前，作用为加快模型训练时的收敛速度，使得模型训练过程更加稳定，避免梯度爆炸或者梯度消失。一定程度上起到正则化作用，几乎替代了Dropout。

  按照channel来计算均值和方差，即可训练参数 β和 γ 的维度等于通道数

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D%26Input%3AB%3D%5C%7Bx_%7B1...m%7D%5C%7D%3B%5Cgamma+%2C%5Cbeta+%28parameters%5C+to%5C+be%5C+learned%29%5C%5C+%26Output%3A%5C%7By_i%3DBN_%7B%5Cgamma+%2C%5Cbeta%7D%28x_i%29%5C%7D+%5Cend%7Balign%7D%5C%5C+%5Cmu_%7BB%7D+%5Cleftarrow%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%7Bx_i%7D%5C%5C+%5Csigma_%7BB%7D%5E2%5Cleftarrow%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28x_i-%5Cmu_B%29%5E2%5C%5C+%5Ctilde%7Bx%7D_i%5Cleftarrow%5Cfrac%7Bx_i-%5Cmu_B%7D%7B%5Csqrt%7B%5Csigma_B%5E2%2B%5Cepsilon%7D%7D%5C%5C+y_i%5Cleftarrow%5Cgamma%5Ctilde%7Bx%7D_i%2B%5Cbeta%5C%5C+)

     归一化的目的：将数据规整到统一区间，减少数据的发散程度，降低网络的学习难度。BN的精髓在于归一之后，使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma+%E3%80%81%5Cbeta) 作为还原参数，在一定程度上保留原数据的分布。 

  ​    对于BN，在训练时，是对每一批的训练数据进行归一化，也即用每一批数据的均值和方差。

  而在测试时，比如进行一个样本的预测，就并没有batch的概念，因此，这个时候用的均值和方差是全量训练数据的均值和方差，这个可以通过移动平均法求得。对于BN，当一个模型训练完成之后，它的所有参数都确定了，包括均值和方差，gamma和bata。

  期望batch较大，这样其分布才更能反映样本的真实分布

  

- Cross-GPU  BN

  跨卡同步BN。当使用多卡时，bn是是计算单卡上的均值和方差。

  先同步求均值，然后各卡再同步求方差。

- Filter Response Normalization （FRN）

  ![img](https://img-blog.csdnimg.cn/20191229120625292.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Jlc3RyaXZlcm4=,size_16,color_FFFFFF,t_70)

   FRN的操作是(H, W)维度上的，即对每个样例的每个channel单独进行归一化，这里 x 就是一个N维度（HxW）的向量，所以FRN没有BN层对batch依赖的问题。  归一化之后同样需要进行缩放和平移变换，这里的![\gamma](https://private.codecogs.com/gif.latex?%5Cgamma)和![\beta](https://private.codecogs.com/gif.latex?%5Cbeta)也是可学习的参数 

   FRN缺少去均值的操作，这可能使得归一化的结果任意地偏移0，如果FRN之后是ReLU激活层，可能产生很多0值，这对于模型训练和性能是不利的。为了解决这个问题，FRN之后采用的阈值化的ReLU，即TLU 。 这里的![\tau](https://private.codecogs.com/gif.latex?%5Ctau)是一个可学习的参数。论文中发现FRN之后采用TLU对于提升性能是至关重要的。 

- Cross-Iteration BN

   论文开门见山的指出 batchsize的大小直接影响了BN的效果，batchsize在小于16后的分类准确率急剧下降。作者在本文提出了一种叫做cross-iteration BN的方法，通过泰勒多项式去估计几个连续batch的统计参数，可以很大程度缓解此问题，在batchsize逐渐变小时，效果依然稳定，并且accuracy始终高于GN的效果。 

### Skip-connections

- #### Residual connections

  ![1604236643771](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604236643771.png)

  因为增加了一项，那么该层网络对x求偏导的时候，多了一个常数项，所以在反向传播过程中，梯度连乘，也不会造成**梯度消失**。

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604237643982.png" alt="1604237643982" style="zoom:50%;" />

  ![1604237628677](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604237628677.png)

- #### Weighted residual connections

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604237670326.png" alt="1604237670326" style="zoom:50%;" />

  ![1604237735789](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604237735789.png)

  引入加权残差结构来解决ReLU和元素加法与深度网络初始化之间的不兼容问题，加权残差网络能够有效地结合不同层次的残差。

- #### Multi-input weighted residual connections

  没找到具体的文献，yolov4中是这样提到的：

  In BiFPN, the multi-input weighted residual connections is
  proposed to execute scale-wise level re-weighting, and then
  add feature maps of different scales.

- #### Cross stage partial connections（CSP）

  来源：https://zhuanlan.zhihu.com/p/116611721
  
   作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的。CSPNet通过将梯度的变化从头到尾地集成到特征图中，在减少了计算量的同时可以保证准确率。CSPNet是一种处理的思想，可以和ResNet、ResNeXt和DenseNet结合。 
  
  - 增强CNN的学习能力，能够在轻量化的同时保持准确性。
  
  - 降低计算瓶颈
  
  - 降低内存成本
  
    <img src="https://pic2.zhimg.com/80/v2-fa5796cecc84cded1f828d32c1189a31_720w.jpg" alt="img" style="zoom: 67%;" />
  
  图中的Transition Layer代表过渡层，主要包含瓶颈层（1x1卷积）和池化层（可选）。（a）图是原始的DenseNet的特征融合方式。（b）图是CSPDenseNet的特征融合方式（trainsition->concatenation->transition）。（c）图是Fusion First的特征融合方式（concatenation->transition）（d）图是Fusion Last的特征融合方式（transition->concatenation）
  
  Fustion First的方式是对两个分支的feature map先进行concatenation操作，这样梯度信息可以被重用。
  
  Fusion Last的方式是对Dense Block所在分支先进性transition操作，然后再进行concatenation， 梯度信息将被截断，因此不会重复使用梯度信息 。
  
  - 使用Fusion First有助于降低计算代价，但是准确率有显著下降。
  - 使用Fusion Last也是极大降低了计算代价，top-1 accuracy仅仅下降了0.1个百分点。
  - 同时使用Fusion First和Fusion Last的CSP所采用的融合方式可以在降低计算代价的同时，提升准确率。

## Bag of freebies

### methods that only change the training strategy or only increase the training cost

### Data augmentation

- photometric distortion

  光度失真

  - brightness

    亮度

  - contrast

    对比度

  - hue

    色调

  - saturation

    色和度

  - noise

    噪声

- geometric distortion

  几何失真

  - random scaling

    随机缩放

  - random cropping

    随机裁剪

  - random flipping

    随机翻转

  - random rotating

    随机旋转

- occlusion

  遮挡

  - random erase

    随机擦除![img](https://img-blog.csdnimg.cn/20190409232211295.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpcWltaW5nMTAw,size_16,color_FFFFFF,t_70)

    

  - CutOut

    和random erase差不多，模仿DropOut，在原图上用固定大小的矩形对图像进行遮挡，类似drop。在矩形范围内，所有的值都被设置为0，或者其他纯色值。而且擦除矩形区域存在一定概率不完全在原图像中的（文中设置为50%） 

    

- multiple images

  ![1604041895967](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604041895967.png)

  - MixUp

    ​    mixup是一种运用在计算机视觉中的对图像进行混类增强的算法，它可以将不同类之间的图像进行混合，从而扩充训练数据集。 

     **数据增强可以提高泛化能力，但这一过程依赖于数据集，而且需要专门知识。其次，数据增强假定领域内样本都是同一类，且没有对不同类不同样本之间领域关系进行建模。** 

     假设batchx1是一个batch样本，batch用是该batch样本对应的标签；batchx2是另一个batch样本，batchy2用是该batch样本对应的标签，λ 是由参数为α，β的**贝塔分布**计算出来的混合系数，由此我们可以得到mixup原理公式为： 

    ![1604030941717](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604030941717.png)

    <img src="https://pic3.zhimg.com/80/v2-a24e855e639eeb4f3a480ba2b6053789_720w.jpg?source=1940ef5c" alt="img" style="zoom:50%;" />

    

  - CutMix

     cutmix就是cut掉一张图片的部分区域，使用另一张图片部分区域进行填充 

  - Mosaic

    四张图片组合

- style transfer GAN

- Self-Adversarial Training

### Semantic distribution bias

- data imbalance

	- hard negative example mining
	- online hard example mining
	- focal loss

### relationship  between different categories

- label smooth

  一般分类标签大多使用独热编码，加上使用的softmax和交叉熵损失的缘故，使得模型训练越来越倾向于使得集中于一个点，忽略了相对的原始分布与样本之间相似的关系，同时鼓励预测值某个类型向1靠近，使得网路预测值要趋于无穷大，逼迫模型去接近真实的标签，有点 “过犹不及” 的意味。

  <img src="https://pic1.zhimg.com/80/v2-56899017cd0d5c113edc8002997381d8_720w.jpg" alt="img" style="zoom: 67%;" />
  
  ![1604374760874](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604374760874.png)
  
   因此标签平滑相当于减少真实标签的类别概率在计算损失值时的权重，同时增加其他类别的预测概率在最终损失函数中的权重。**这样真实类别概率和其他类别的概率均值之间的gap（倍数）就会下降一些，降低模型过度自信，提升模型的泛华能力。** 

- knowledge distillation

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604371474569.png" alt="1604371474569" style="zoom:80%;" />

  ​        知识蒸馏（Knowledge Distilling）是模型压缩的一种方法，是指利用已经训练的一个较复杂的Teacher模型，指导一个较轻量的Student模型训练，从而在减小模型大小和计算资源的同时，尽量保持原Teacher模型的准确率的方法。 

  ​    **Knowledge Distill是一种简单弥补分类问题监督信号不足的办法。** 

  ​        **KD的核心思想在于"打散"原来压缩到了一个点的监督信息，让student模型的输出尽量match teacher模型的输出分布。**其实要达到这个目标其实不一定使用teacher model，在数据标注或者采集的时候本身保留的不确定信息也可以帮助模型的训练 

  ​        具体步骤为：

  1、训练大模型：先用hard target，也就是正常的label训练大模型。
  2、计算soft target：利用训练好的大模型来计算soft target。也就是大模型“软化后”再经过softmax的output。
  3、训练小模型，在小模型的基础上再加一个额外的soft target的loss function，通过lambda来调节两个loss functions的比重。（原本硬标签loss，加上另一个软标签的分支，分支温度和大模型一致，调整这两个loss分支对总loss的贡献）
  4、预测时，将训练好的小模型按常规方式（右图）使用。

### objective function of BBox regression

​    The traditional object detector usually uses Mean Square Error (MSE) to directly perform regression on the center point coordinates and height and width of the BBox. However, to directly estimate the coordinate values of each point of the BBox is to treat these points as independent variables, but in fact does not consider the integrity of the object itself.

​     IoU loss 是通过BBox 与 GT 的坐标算得的IoU，为尺度不变的表示，真实反映了实际的重合度，而不与物体大小有关。但是用坐标偏移加均方差的计算方式对尺度敏感。

**好的目标框回归损失应该考虑三个因素：重叠面积、中心点距离、长宽比**

来源： https://zhuanlan.zhihu.com/p/139724869?utm_source=wechat_session

- #### MSE

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604285491373.png" alt="1604285491373" style="zoom:80%;" />

- #### IoU loss

  ![img](https://pic1.zhimg.com/80/v2-2f07c3a96c0e1c5590e620c4966e78dc_720w.jpg)

  ![[公式]](https://www.zhihu.com/equation?tex=IoU+loss+%3D+-ln%28IoU%29)

  ​																	或者![[公式]](https://www.zhihu.com/equation?tex=IoU+loss+%3D+1-IoU)

IoU loss存在两个不足

- 预测值和Ground truth没有重叠的话，IoU始终为0且无法优化
- IoU无法辨别不同方式的对齐，比如方向不一致等

![img](https://pic3.zhimg.com/80/v2-933c315fb35be06128367187d589d5fe_720w.jpg)

- #### GIoU loss

  ![[公式]](https://www.zhihu.com/equation?tex=GIoU%3DIoU-%5Cfrac%7B%E5%B9%B6%E9%9B%86%E5%A4%96%E6%8E%A5%E7%9F%A9%E5%BD%A2%E7%A9%BA%E9%9A%99%E9%9D%A2%E7%A7%AF%7D%7B%E5%B9%B6%E9%9B%86%E5%A4%96%E6%8E%A5%E7%9F%A9%E5%BD%A2%E9%9D%A2%E7%A7%AF%7D%3DIoU-%5Cfrac%7BC%5Cbackslash%28A%5Ccup%7BB%7D%29%7D%7B%7CC%7C%7D)

  ![[公式]](https://www.zhihu.com/equation?tex=GIou+loss+%3D+1-GIoU)

  尽管GIoU解决了IoU的梯度问题，但仍然存在几个不足：

  - 当GT完全包裹BBox的时候，IoU和GIoU的值都一样，此时GIoU退化为IoU, 无法区分其相对位置关系
  - 在训练过程中，GIoU倾向于先增大bbox的大小来增大与GT的交集，然后通过IoU项引导最大化bbox的重叠区域。
  - 由于很大程度依赖IoU项，GIoU需要更多的迭代次数来收敛

- #### DIoU loss

  基于IoU和GIoU存在的问题，DIoU提出了两个问题：

  - 直接最小化预测框与目标框之间的归一化距离是否可行，以达到更快的收敛速度。
  - 如何使回归在与目标框有重叠甚至包含时更准确、更快。

  ​    基于问题一，DIoU Loss相对于GIoU Loss收敛速度更快，考虑了重叠面积和中心点距离，但没有考虑到长宽比； 

  <img src="C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604288421861.png" alt="1604288421861" style="zoom:80%;" />

  ![1604288600134](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604288600134.png)

   DIoU可以用于NMS，不仅考虑重叠区域，还考虑了中心点距离。 

  ![1604289033227](C:\Users\BigHuang\AppData\Roaming\Typora\typora-user-images\1604289033227.png)

   其中 ![[公式]](https://www.zhihu.com/equation?tex=s_%7Bi%7D) 是分类置信度， ![[公式]](https://www.zhihu.com/equation?tex=%5Cvarepsilon) 为NMS阈值，M为最高置信度的框。DIoU-NMS倾向于中心点距离较远的box为不同的对象。 

- #### CIoU loss

   CIoU的惩罚项是在DIoU的惩罚项基础上加了一个影响因子 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%5Cupsilon) 。其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是权重系数， ![[公式]](https://www.zhihu.com/equation?tex=%5Cupsilon) 度量长宽比的相似性。 

![[公式]](https://www.zhihu.com/equation?tex=%5Cupsilon%3D%5Cfrac%7B4%7D%7B%5Cpi%5E2%7D%5Cleft%28arctan%5Cfrac%7Bw%5E%7Bgt%7D%7D%7Bh%5E%7Bgt%7D%7D-arctan%5Cfrac%7Bw%7D%7Bh%7D%5Cright%29%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%3D%5Cfrac%7B%5Cupsilon%7D%7B%281-IoU%29%2B%5Cupsilon%7D)

![[公式]](https://www.zhihu.com/equation?tex=CIoU%3DDIoU-%5Calpha%5Cupsilon)

![[公式]](https://www.zhihu.com/equation?tex=CIoU+loss%3D1-CIoU)





