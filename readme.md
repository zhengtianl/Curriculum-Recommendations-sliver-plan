Kaggle Learning Equality - Curriculum Recommendations
学生课程推荐竞赛银牌算法概览

所用算法：
1.本解决方案基于检索与重排的方式，根据主题内容来给学生推荐相关课程。
2.采用迁移学习方法进行语义检索，首先对预训练模型xlm-roberta-large使用MNRL（Multiple Negatives Ranking Loss）损失函数进行训练，将训练所得权重用来提取主题（topic）和内容（content）的嵌入（embedding）。
3.使用KNN聚类算法获取每个主题top K个的内容（content），即最相关的前K个内容。
4.使用分类模型将此K个内容进行重排（ReRank），通过预测每个内容的相关分数，判断其是否属于正确匹配项。此时分类模型主干（backbone）所用权值继承了第2步中模型的权值。
5.最后根据验证集设置阈值得到最后的预测结果。

预训练模型：
xlm-roberta-large 