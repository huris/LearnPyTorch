# 如何做好一个ML问题的分析

### 第一部分： 定义问题
分析题目，弄明白3个问题：

- 题目要求做什么？ 
- 采用什么样的方式？
- 最后的评估方式是什么？ 

明白上述3个问题后，就可以：

* 导入相应的Python模块和包
* 导入数据集
### 第二部分：理解数据
>  加强对数据理解的步骤 ， 通过描述性统计和可视化数据信息
 * 描述性统计：查看数据基本格式，主要包括：
      * .shape：数据的形状
     * . head()：数据本身
     * .info()：数据集的简单描述，总行数，空值或者数据类型等
     * .value_counts()：查看类别个数（分类的时候用的多）
    * .describe()：简单的统计性表述，最大/最小值，均值等
     * . corr(method='pearson')：列之间的相关性
     * .skew()：分析数据的高斯分布来确认数据的偏离情况 
    
* 可视化数据信息
  * 绘制每个属性的直方图（直观展示每个属性分布情况：高斯分布，指数分布还是偏态分布），由一系列高度不等的纵向条纹或者线段表示数据的分布情况。
  * 横轴表示数据类型，纵轴表示分布情况。
  * seaborn是一个比matplotlib更好用的工具，画出的图像也更加好看。
  
* 数据EDA（探测性分析，生成数据报告）
  导入pandas_profiling包，调用函数可以直接生成数据的探测性报告，包括上面的描述性统计和可视化信息，相关性等。

  **很重要，只有采用正确的可视化方法，才能真正去理解数据，为之后的相应处理打下基础**

```python
import pandas_profiling as ppf
ppf.ProfileReport(data)
```
### 第三部分：数据准备
* 数据清洗与转换：对数据进行重新审查和校验，删除重复信息，纠正存在的错误，提供数据一致性
>难点： 数据清洗一般针对具体应用， 因而难以归纳统一的方法和步骤，但根据数据不同可以给出相应的数据清理方法：
>>1. 解决缺失值： 平均值、最大值、最小值或更为复杂的概率估计代替缺失值. PS: 缺失值的处理，要分数据类型，字符型，整型等不同的类型处理填充的方式不一样
>>2. 去重：相等的记录合并为一条记录（即合并/清除）
>>3. 解决错误值：用统计分析方法识别可能的错误值或异常值，如偏差分析、识别不遵守分布或回归方程的值，也可以用简单规则库（常识性规则，业务规则等）检查数据值，或使用不同属性间的约束，外部的数据来检测和清理数据。 PS : 这个需要结合可视化，删除一些重复或者异常的点，把偏斜度大的特征进行boxcox转换成标准的正态
>>4. 解决数据的不一致性： 比如数据是类别型（LabelEncoder或者OneHotEncoder）或者次序型
>>
> 数据清洗的八大场景：
>>1. 删除多列
>>2. 更改数据类型
>>3. 将分类变量转换为数字变量
>>4. 检查缺失数据(一般是NAN)
>>5. 删除列中的字符串
>>6. 删除列中的空格
>>7. 用字符串连接两列（带条件）
>>8. 转换时间戳（从字符串到日期时间格式）
>>
> 数据处理方法：
>> 1. 对数变换(log1p)
>> 2. 标准缩放(StandardScaler)
>> 3. 转换数据类型(astype)
>> 4. 独热编码(OneHotEncoder或者pd.get_dummies)
>> 5. 标签编码(LabelEncoder)
>> 6. 修复偏斜特征(boxcox1p)
* 特征工程（很重要）
  根据前面分析的特征与特征之间的相关性，特征与目标变量之间的相关性等对特征进行选择，组合，删除。

  可以基于特征重要性来选择最相关的特征，或者进行各种组合。

这部分处理完之后，分离出训练集和测试集，进行后续的创建模型，训练和预测。

 数据处理的好坏，决定模型的表现。

### 第四部分：评估算法
寻找最佳的算法子集，包括：

- 分离评估数据集，便于验证模型
- 定义模型评估标准，评估算法模型
- 抽样审查线性算法和非线性算法，比较算法的准确度

分离数据集：随机划分/随机抽样/分层抽样等。

评估算法：一般情况建立多个可以处理相应问题的模型，进行筛选， 通过model字典的方式进行。

```python
models = {}

models['LR'] = LinearRegression()
models['Ridge'] = Ridge()
models['Lasso'] = Lasso(alpha=0.01,max_iter=10000)
models['RF'] = RandomForestRegressor()
models['GBR'] = GradientBoostingRegressor()
models['LinSVR'] = LinearSVR()
models['SGD'] = SGDRegressor(max_iter=1000,tol=1e-3)
models['Extra'] = ExtraTreesRegressor()
models['Xgb'] = XGBRegressor(n_estimators=400)
models['lgb'] = LGBMRegressor()

# 评估算法
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=7)   # Kfold 交叉验证函数
    cv_results = np.sqrt(-cross_val_score(models[key], X_train_scaled, y_log, cv=kfold, scoring='neg_mean_squared_error'))
    results.append(cv_results)
    print('%s: %f (%f)' %(key, cv_results.mean(), cv_results.std()))
```

```python
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
```
### 第五部分：优化模型
得到一个准确度足够的算法列表之后， 从中找出最合适的算法，对每一种算法进行调参， 得到最佳结果，使用集成算法来提高算法模型的准确度。

有两种优化模型的方式：
* 算法调参：从上面表现比较好的模型中选出2-3个，可以尝试进行合理的参数搜索，找到合适的参数：

```cpp
cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

model = XGBRegressor(**other_params)
grid = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
grid.fit(X_train_scaled, y_log)

print("最高得分：%.3f" % grid.best_score_)
print("最优参数: %s" % grid.best_params_)
```
这是XGB用网格搜索合适参数的模板。
* 集成算法：仅用一种算法往往效果不好，需要几种模型一起集成或者堆叠等，XGB，lightgbm，Adaboost等都是直接封装好的集成方法，但现在在这个基础上又有一些新的集成和堆叠或者模型融合的技术。
基本的集成技术如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118095758548.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
模型的堆叠
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118095830113.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d1emhvbmdxaWFuZw==,size_16,color_FFFFFF,t_70)
这里给出一个模型融合的模板（来自于房价预测)：

```cpp
# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)
```

```cpp
kf = KFold(n_splits=12, random_state=42, shuffle=True)
# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)


# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006,random_state=42)

# StackingCVRegressor 
stackReg = StackingCVRegressor(regressors=(xgboost, svr, ridge, gbr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True,random_state=42)
```
上面的模型建好了之后，要记得先fit(X, y)
```cpp
def blended_predictions(X,weight):
    return ((weight[0] * ridge_model_full_data.predict(X)) + \
            (weight[1] * svr_model_full_data.predict(X)) + \
            (weight[2] * gbr_model_full_data.predict(X)) + \
            (weight[3] * xgb_model_full_data.predict(X)) + \
            (weight[4] * lgb_model_full_data.predict(X)) + \
            (weight[5] * stack_reg_model.predict(np.array(X))))
```
下面调用评分
```cpp
blended_score = rmsle(train_labels, blended_predictions(X,[0.15,0.2,0.1,0.15,0.1,0.3]))
print("blended score: {:.4f}".format(blended_score))
model_score['blended_model'] =  blended_score
```
### 第六部分：验证模型
选取出合适的模型（集成的或者堆叠的），带入测试集，进行预测得出结果，根据比赛要求的格式进行文件的保存，提交。

### 结论：
以上，就是对ML问题处理的思路，经过六个步骤可以让问题处理变得更清晰一些，可能不适用于所有人，并且上面六个部分也不一定严格的按照这个顺序来，活学活用，处理方式也可以多变，这里只给出一个思路框架。
有了处理问题的整体框架，再学习一些处理问题的方法，多练多看，相信自己也会处理的很好。

