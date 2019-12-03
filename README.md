### 不做任何特征处理，只做调参后的得分。
extra\_tree验证集上的均方误差为0.1106;5折交叉验证0.1265，线上得分0.1299
rf验证集上的均方误差为0.1160;5折交叉验证0.1284，线上得分0.1397
gbdt验证集上的均方误差为0.0968;5折交叉验证0.1137，线上得分0.1437
xgboost验证集上的均方误差为0.1006;5折交叉验证0.1152，线上得分0.1481
lightgbm验证集上的均方误差为0.0976,5折交叉验证0.1121，线上得分0.1489

adaboost验证集上的均方误差为0.1621;5折交叉验证0.1604，线上得分0.1770
svm验证集上的均方误差为0.0928;5折交叉验证0.1101，线上得分0.9763。

1. 分析结果：验证集的结果比交叉验证的结果更好，原因可能是验证集时使用全部的训练集，而每次交叉验证只用了80%的训练集。

### 重新调整超参数的范围，只做调参后的得分。
1. lightgbm交叉验证结果:-0.1132,验证集上结果:0.0989,线上得分:0.1529
2. xgboost交叉验证结果:-0.1180,验证集上结果:0.0938,线上得分:0.1575
3. gbdt交叉验证结果:-0.1149,验证集上结果:0.1026,线上得分:0.1513
4. extra_tree交叉验证结果:-0.1268,验证集上结果:0.1102,线上得分:0.1292
5. model_ensemble验证集上结果:0.0937，线上得分:0.1388。

6. 分析结果：调整范围后，结果反而更差了，可能是因为超参数的搜索范围变大了。这也说明超参数的选择对模型的影响很大，后期需要更精细地调整超参数。 


### 特征处理方法
1. 训练集和测试集并不同分布，将特征在train和test上的分布，展示出来。删除分布差异大的特征。
2. 使用基模型训练后，删除拟合效果差的样本。

### 需要做的任务
1. 阅读并运行别人的代码，总结其有效的，可重复使用的特征处理方法。
	1. 记录所有出现的方法
	2. 对每种方法按照自己的理解解释其原理。
	3. 单独运行每一种方法，对比使用方法前和使用后的提升区别。
	4. 由于需要控制模型的差异给特征处理实验产生的影响。模型统一使用extra_tree(范化能力比较强，作为单模型跑出来的结果也最好)。
2. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.21.3f9274ffv58xbx&postId=41563 
	1. 先在天池的kernel上运行一遍，确保运行成功。<部分完成，最后几个模型跑的时候数据有问题，不管了。>
	2. 展示所有特征在训练集和测试集上的特征分布，然后删除分布不一致的特征。<看完了，未实验和总结代码>
	3. 使用模型拟合后，查看残差大于3sigma的样本点。作为异常点删除。思考是对所有的模型进行，还是只有这个模型这样做。<看完了，未实验和总结代码>
	
3. https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12282029.0.0.6a8ffa50ySlL3U&postId=60069

4. tensorflow2.0的包 https://files.pythonhosted.org/packages/a1/eb/bc0784af18f612838f90419cf4805c37c20ddb957f5ffe0c42144562dcfa/tensorflow_gpu-2.0.0-cp37-cp37m-manylinux2010_x86_64.whl
5. https://files.pythonhosted.org/packages/82/c0/371cf368e2d8b1b7bcf9f9bafd7cec962487e654ad8296d8e0ad62011537/protobuf-3.11.0-cp37-cp37m-manylinux1_x86_64.whl

