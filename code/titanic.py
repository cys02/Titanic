# -*- coding：utf-8 -*-
import time

import matplotlib
import numpy as np
import pandas as pd
# 载入数据可视化分析工具库Seaborn、 Matplotlib；
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from fpgrowth_py import fpgrowth

sns.set(style='white', context='notebook', palette='deep')

pd.set_option('display.max_columns', None)  # 展示所有列
pd.set_option('display.width', 180)  # 设置宽度

#------------------------------------------------------------------
#1.下载和导入数据
#------------------------------------------------------------------

#使用训练集和测试集，使用 Pandas的 read_csv() 方法将数据读入 DataFrame
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#使用shape属性分别查看训练集和数据集的规模
# print('训练集规模：',train.shape)
# print('测试集规模：',test.shape)

#使用 DataFrame的 append()方法将两个数据集合并，以便进行数据清洗。
data = train.append(test,ignore_index=True)

#------------------------------------------------------------------
#2.数据清洗
#------------------------------------------------------------------

#使用DataFrame的 head()、 describe()、 info()等方法查看数据样例、 数据类型 、 缺失情况获取数据项的均值、中位数、最大最小值、标准差；使用
print("train数据样例：")
print(train.head())
print("train数据描述：")
print(train.describe())
print("train数据信息：")
print(train.info())
print("test数据信息：")
print(test.info())

#使用 heatmap()方法对缺失值情况进行可视化
#1.训练集缺失值可视化
plt.figure(figsize=(20,14))
sns.heatmap(train.isnull(),cmap='gray')
plt.title('train data missing value')
plt.show()

#2.测试集缺失值可视化
plt.figure(figsize=(20,14))
sns.heatmap(test.isnull(),cmap='gray')
plt.title('test data missing value')
plt.show()

#Fare缺失值采用平局数填充
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

#Embarked缺失值采用众数填充
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

#Cabine缺失值采用Unknown填充
data['Cabin'] = data['Cabin'].fillna('Unknown')

#使用 train_test_split()方法 将特征集合重新划分为训练集和测试集
train_data = data[:891]
test_data = data[891:]

#Age

#Age缺失值采用随机森林填充
from sklearn.ensemble import RandomForestRegressor
age_df = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
#将已有的数值型特征取出来丢进Random Forest Regressor中
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]
Y = age_df_notnull.values[:,0]
RFR = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
RFR.fit(X,Y)
predictAges = RFR.predict(age_df_isnull.values[:,1:])
train_data.loc[(train_data['Age'].isnull()),'Age'] = predictAges

#显示缺失情况
print("train_data数据信息：")
print(train_data.info())

#------------------------------------------------------------------
#3.可视化分析
#------------------------------------------------------------------

#分别画出train_data中survived=0和survived=1的年龄分布图
fig,ax = plt.subplots(1,2,figsize=(18,8))
train_data['Age'][train_data['Survived']==0].plot.hist(ax=ax[0],bins=20,edgecolor='black',color='cornflowerblue')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_data['Age'][train_data['Survived']==1].plot.hist(ax=ax[1],bins=20,edgecolor='black',color='cornflowerblue')
ax[1].set_title('Survived=1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()

#在一张图画出survived=0和survived=1随年龄的变化情况
#增加图像大小
train_data.Age[train_data.Survived == 0].plot(kind='kde')
train_data.Age[train_data.Survived == 1].plot(kind='kde')
plt.xticks(range(-20, 125, 20))
plt.legend(['died', 'survived'])
plt.title('Survival Rate per Age', fontproperties='SimHei')
plt.show()

# 画出不同年龄段survived=0和survived=1的对比柱状图
newdata = train_data
newdata["Age"] = newdata["Age"].astype(np.int32)
bins = np.arange(0, 85, 10)
count_bins = pd.cut(newdata["Age"], bins)
age_data = newdata.groupby([count_bins, "Survived"]).count()["Name"]
newdata["count_bins"] = count_bins.values
sns.countplot(x="count_bins", hue="Survived", data=newdata)
plt.legend(['died', 'survived'])
plt.show()

#不同年龄下的生存率
fig,axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Age_int'] = train_data['Age'].astype(int)
average_age = train_data[['Age_int','Survived']].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age)
plt.tight_layout()
plt.show()

#性别与生存率的关系
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color='cornflowerblue')
plt.title('性别与生存率的关系',fontproperties='SimHei')
plt.tight_layout()
plt.show()

#船舱等级与生存率的关系
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(color='cornflowerblue')
plt.title('船舱等级与生存率的关系',fontproperties='SimHei')
plt.tight_layout()
plt.show()

#不同等级船舱的男女生存率
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex'])\
    .mean().plot.bar(color='cornflowerblue')
plt.title('不同等级船舱的男女生存率',fontproperties='SimHei')
plt.tight_layout()
plt.show()

#查看不同称呼和性别的关系
#将名字中的逗号和句号之间的内容提取出来
train_data['Title'] = train_data['Name']\
    .str.extract('([A-Za-z]+)\.',expand=False)
#输出Title的取值和Sex的取值之间的关系
print(pd.crosstab(train_data['Title'],train_data['Sex']))

#不同称呼和幸存人数的关系
sns.countplot(x="Title", hue="Survived", data=train_data)
plt.title('不同称呼和幸存人数的关系',fontproperties='SimHei')
plt.xticks(rotation=60)
plt.tight_layout()
plt.show()


#查看不同称呼的生存率的关系
train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()
plt.title('不同称呼和生存率的关系',fontproperties='SimHei')
plt.tight_layout()
plt.show()

#亲友人数和存活与否的关系
fig1, axis1=plt.subplots(1,1,figsize=(15,5))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar()
plt.title('泰坦尼克号上的父母/子女数量与生存率的关系',fontproperties='SimHei')
plt.show()

fig2, axis2=plt.subplots(1,1,figsize=(15,5))
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar()
plt.title('泰坦尼克号上的兄弟姐妹/配偶数量与生存率的关系',fontproperties='SimHei')
plt.show()

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp']+1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()
plt.title('泰坦尼克号上的家庭人数与生存率的关系',fontproperties='SimHei')
plt.show()

#票价分布和存活与否的关系
plt.figure(figsize=(10, 5))
train_data['Fare'].hist(bins=70)
plt.title('票价分布',fontproperties='SimHei')
plt.show()

train_data.boxplot(column='Fare', by='Pclass', showfliers=False)
plt.title('不同等级船舱的票价分布',fontproperties='SimHei')
plt.show()

#港口和存活与否的关系
sns.countplot(x="Embarked", hue="Survived", data=train_data)
plt.title('不同港口上船的人数和幸存人数的关系',fontproperties='SimHei')
plt.show()

#不同港口上船的人数和存活率的关系
train_data[['Embarked','Survived']].groupby(['Embarked']).mean().plot.line()
plt.title('不同港口上船的人数和存活率的关系',fontproperties='SimHei')
plt.show()

#------------------------------------------------------------------
# 4.特征提取
#------------------------------------------------------------------

#使用pd.get_dummies()函数将登船港口类别型特征转换为数值型特征
#对登船港口进行one-hot编码
embark_dummies = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
train_data.drop(['Embarked'], axis=1, inplace=True)

embark_dummies = train_data[['S','C','Q']]
print(embark_dummies.head())

#将train_data中的Sex列中的male和female转换为数值型特征
train_data['Sex'] = train_data['Sex']\
    .apply(lambda x: 1 if x == 'male' else 0)  # 性别替换男为1,女为0

#对Fare列进行归一化处理
train_data['Fare'] = train_data['Fare'].\
    apply(lambda x: (x - train_data.Fare.min())
                    / (train_data.Fare.max() - train_data.Fare.min()))

#头衔映射
titledict = {
    "Capt": "officer",
    "Col": "officer",
    "Major": "officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "officer",
    "Rev": "officer",
    "Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}
train_data.Title = train_data.Title.map(titledict)  # 头衔映射
#对映射后的头衔进行one-hot编码
title_dummies = pd.get_dummies(train_data['Title'], prefix='Title')
train_data = pd.concat([train_data, title_dummies], axis=1)
train_data.drop('Title', axis=1, inplace=True)

title_dummies = train_data[['Title_Master'
    , 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Royalty']]
print(title_dummies.head())

#------------------------------------------------------------------
# 5.特征选择
#------------------------------------------------------------------

#在进行特征工程的时候，我们不仅需要对训练数据进行处理，
# 还需要同时将测试数据同训练数据一起处理，使得二者具有相同的数据类型和数据分布。
train_df_org = pd.read_csv('train.csv')
test_df_org = pd.read_csv('test.csv')
#填充测试集中Survived的缺失值
test_df_org['Survived'] = 0
#将训练集和测试集合并
combined_train_test = train_df_org.append(test_df_org,ignore_index=True)


# 5.1 登船港口Embarked
#因为“Embarked”项的缺失值不多，所以这里我们以众数来填充：
combined_train_test['Embarked']\
    .fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
#将Embarked特征进行分类
combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
#使用pd.get_dummies获取one-hot编码
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked']
                                , prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

# 5.2 性别Sex
# 为了后面的特征分析，将Sex特征进行分类
combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
#获取Sex的one-hot编码
sex_dummies_df = pd.get_dummies(combined_train_test['Sex']
                                , prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

#5.3 名字Name
#首先从名字中提取各种称呼
combined_train_test['Title'] = combined_train_test['Name'].str.extract('([A-Za-z]+)\.',expand=False)
#头衔映射
titledict = {
    "Capt": "officer",
    "Col": "officer",
    "Major": "officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "officer",
    "Rev": "officer",
    "Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}
combined_train_test.Title = combined_train_test.Title.map(titledict)  # 头衔映射

#将Title特征进行分类
#进行one-hot编码
combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
title_dummies_df = pd.get_dummies(combined_train_test['Title']
                                  ,prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test,title_dummies_df],axis=1)

#增加名字长度的特征
combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)

# 5.4 船票价格Fare
#对于Fare项，由于缺失值只有一个，所以我们直接用平均数来填充：
combined_train_test['Fare']\
    .fillna(combined_train_test['Fare'].mean(), inplace=True)
#通过对Ticket数据的分析，我们可以看到部分票号数据有重复，同时结合亲属人数及名字的数据，和票价船舱等级对比，
# 我们可以知道购买的票中有家庭票和团体票，所以我们需要将团体票的票价分配到每个人的头上
combined_train_test['Group_Ticket'] = combined_train_test['Fare']\
    .groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'],axis=1,inplace=True)

#给票价分段
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'],5)
#对分段后的票价进行分类
combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(combined_train_test['Fare_bin_id'])\
    .rename(columns=lambda x: 'Fare_' + str(x))
combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_df], axis=1)
combined_train_test.drop(['Fare_bin'],axis=1, inplace=True)

# 5.4.1 查看不同等级船舱与票价的关系
print("查看不同等级船舱与票价的关系")
print(combined_train_test['Fare']
      .groupby(by=combined_train_test['Pclass']).mean())

# 5.6 家庭人数Family_Size
#将二者合并为FamliySize这一组合项
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'

combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category)

le_family = LabelEncoder()
le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],
                                        prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)

# 5.7 年龄Age
#对于Age项，我们可以看到有大量的缺失值，我们可以通过一些特征来推测年龄，比如说头衔，船舱等级，家庭人数等等
#使用机器学习算法来预测Age
#以Age为目标值，将Age完整的项作为训练集，将Age缺失的项作为测试集
#构造训练集和测试集
missing_age_df = pd.DataFrame(combined_train_test[
    ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
     'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
#展示测试集
print("展示年龄属性测试集")
print(missing_age_test.head())
#建立Age的预测模型，采用多模型预测，然后再做模型的融合，提高预测的精度。
#下面这个函数
def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # 这段代码做了以下几件事情：
    # 首先，它定义了一个GBM回归器，并指定了随机种子为42。
    # 然后，它定义了一个参数网格，用于在训练过程中对模型的参数进行交叉验证。
    # 接着，它使用GridSearchCV函数来选择最优参数。这个函数使用交叉验证来评估不同参数组合的模型的性能，并选择性能最优的一组参数。
    # 接下来，它使用训练数据训练GBM回归器，并使用训练好的模型来预测测试集中的年龄，并将这些预测值添加到测试数据集的'Age_GB'列中。
    # 最后，它打印出测试集中前四个预测的年龄值。

    #采用GBM模型
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4]
        , 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg
        , gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('年龄特征最佳 GB 参数:' + str(gbm_reg_grid.best_params_))
    print('年龄特征最佳 GB 得分:' + str(gbm_reg_grid.best_score_))
    print('“年龄”特征回归器的 GB 训练误差:' + str(
        gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print("GBM模型预测信息如下：")
    print(missing_age_test['Age_GB'][:4])

    # 这段代码做了以下几件事情：
    # 首先，它定义了一个随机森林回归器。
    # 然后，它定义了一个参数网格，用于在训练过程中对模型的参数进行交叉验证。
    # 接着，它使用GridSearchCV函数来选择最优参数。这个函数使用交叉验证来评估不同参数组合的模型的性能，并选择性能最优的一组参数。
    # 接下来，它使用训练数据训练随机森林回归器，并使用训练好的模型来预测测试集中的年龄，并将这些预测值添加到测试数据集的'Age_RF'列中。
    # 最后，它打印出测试集中前四个预测的年龄值。

    #采用RF模型
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                  scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('年龄特征最佳RF参数:' + str(rf_reg_grid.best_params_))
    print('年龄特征最佳RF分数:' + str(rf_reg_grid.best_score_))
    print('“年龄”特征回归器的 RF 训练误差:' + str(
        rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print("RF模型预测信息如下：")
    print(missing_age_test['Age_RF'][:4])

    #首先，它打印了测试集中'Age'列和'Age_GB'、'Age_RF'列的形状。
    #然后，它使用测试集中'Age_GB'列和'Age_RF'列的平均值来更新测试集中的'Age'列。
    #最后，它从测试集中删除'Age_GB'和'Age_RF'列。

    #两个模型合并
    print('行数和列数:', missing_age_test['Age'].shape, missing_age_test[['Age_GB'
                                                , 'Age_RF']].mode(axis=1).shape)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB']
                                                 , missing_age_test['Age_RF']])
    print("合并后的预测信息如下：")
    print(missing_age_test['Age'][:4])
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
    return missing_age_test

#对缺失的年龄进行预测
combined_train_test.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
# #展示填充后的数据
# print(missing_age_test.head())

#5.8 船票Ticket
#观察Ticket的值，可以看到，Ticket有字母和数字之分，而对于不同的字母，
# 可能在很大程度上就意味着船舱等级或者不同船舱的位置，也会对Survived产生一定的影响，
# 所以将Ticket中的字母分开，为数字的部分则分为一类。
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0]
combined_train_test['Ticket_Letter'] = combined_train_test['Ticket_Letter']\
    .apply(lambda x: np.NaN if x.isnumeric() else x)

#使用factorize将Ticket_Letter分为不同的类别
combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

#5.9船舱Cabin
# 因为Cabin项的缺失值确实太多了，我们很难对其进行分析，或者预测。
# 所以这里我们可以直接将Cabin这一项特征去除。但通过上面的分析，可以知道，
# 该特征信息的有无也与生存率有一定的关系，所以这里我们暂时保留该特征，并将其分为有和无两类。
combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
combined_train_test['Cabin'] = combined_train_test['Cabin']\
    .apply(lambda x: 0 if x == 'U0' else 1)

#5.10 特征间相关性分析
#挑选一些主要的特征，生成特征之间的关联图，查看特征与特征之间的相关性
Correlation = pd.DataFrame(combined_train_test[['Embarked','Sex','Title','Name_length','Family_Size',
                                                'Family_Size_Category','Fare','Fare_bin_id','Pclass'
                                                ,'Age','Ticket_Letter','Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correaltion of Feature',y=1.05,size=15)
sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True
            ,cmap=colormap,linecolor='white',annot=True)
plt.tight_layout()
plt.show()

#方法计算特征与是否幸存的相关性；使用 sklearn工具库中的ExtraTreesClassifier类计算特征的重要性；
y = combined_train_test['Survived'].loc[0:890]
X = combined_train_test.loc[:890, ['Title', 'Sex', 'Pclass', 'Fare', 'Family_Size', 'Embarked', 'Age']]
extra_tree_forest = ExtraTreesClassifier(n_estimators=8, criterion='entropy', max_features=4)
extra_tree_forest.fit(X, y)

# 计算每个特征的重要性水平
feature_importance = extra_tree_forest.feature_importances_
# 标准化特征的重要性水平
feature_importance_normalized = np.std(
    [tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis=0)
myfont = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\STKAITI.TTF')
# 画图
plt.barh(X.columns, feature_importance_normalized)
plt.xlabel('Feature Importance', fontproperties=myfont)
plt.ylabel('Feature', fontproperties=myfont)
plt.title('Comparison of Feature Importances', fontproperties=myfont)
plt.tight_layout()
plt.show()


#选取combined_train_test中的前12列
df_first_12_columns = combined_train_test
df_first_12_columns=df_first_12_columns.iloc[:, :12]
cor = df_first_12_columns.corr(numeric_only=True)
sns.heatmap(cor, annot=True, annot_kws={'size': 10})
plt.tight_layout()
plt.show()

cor.drop(['Survived'], inplace=True)
cor = cor.sort_values(by='Survived')
p1 = plt.barh(cor.index, cor['Survived'])
plt.bar_label(p1, fmt='%0.04g', label_type='center')
plt.title('Titanic Survival - Correlation')
plt.tight_layout()
plt.show()


#保留年龄、性别、客舱等级、登船港口、头衔、船票价格、家庭人数等特征
new_combined_data = combined_train_test
new_combined_data = new_combined_data.loc[:,
        ['Survived','Age','Sex','Pclass','Embarked', 'Title', 'Fare', 'Family_Size']]

#------------------------------------------------------------------
#6. 关联规则分析
#------------------------------------------------------------------

#6.1加载工具库、进一步处理特征
new_combined_data["AgeClass"] = pd.cut(new_combined_data.Age
                            , bins=[0, 4, 10, 20, 40, 60, 100], right=False,
                            labels=['Infant', 'Child', 'Teenager', 'YoungAdult', 'Adult', 'Elderly'])
age1 = new_combined_data['AgeClass']
age1 = age1.head(891)
age2 = new_combined_data['AgeClass'].tail(417)
counts1 = age1.value_counts()
counts2 = age2.value_counts()

plt.bar(counts1.index, counts1.values)
for i, v in enumerate(counts1.values):
    plt.text(i, v, str(v))
plt.title('Train-AgeClass')
plt.show()

plt.bar(counts2.index, counts2.values)
for i, v in enumerate(counts2.values):
    plt.text(i, v, str(v))
plt.title('Test-AgeClass')
plt.show()

new_combined_data=new_combined_data.loc[:890, :]
new_combined_data = new_combined_data.astype(str)

#将new_combined_data中的数字类型的数据重新映射回字符串

# 性别特征
sexdict = {
    '0': "female",
    '1': "male"
}
new_combined_data.Sex = new_combined_data.Sex.map(sexdict)  # 客舱等级映射
new_combined_data['Sex'] = new_combined_data['Sex'].astype('category')

# 头衔特征
titledict = {
    '0': 'Mr',
    '1': 'Mrs',
    '2': 'Miss',
    '3': 'Master',
    '4': 'Royalty',
    '5': 'officer'
}
new_combined_data['Title'] = new_combined_data.Title.map(titledict)  # 头衔类别映射
new_combined_data['Title'] = new_combined_data['Title'].astype('category')

new_combined_data['FamilySize'] = new_combined_data['Family_Size'].astype('category')

#生存情况映射
surviveddict = {
    '0': 'Dead',
    '1': 'Survived'
}
new_combined_data['Survived'] = new_combined_data.Survived.map(surviveddict)  # 头衔类别映射
new_combined_data['Survived'] = new_combined_data['Survived'].astype('category')

embarkeddict = {
    '0': "S",
    '1': "C",
    '2': "Q"
}
new_combined_data.Embarked = new_combined_data.Embarked.map(embarkeddict)  # 登船港口映射

#将剩余属性转换为字符串
new_combined_data = new_combined_data.astype(str)

df_array = new_combined_data.to_numpy()
df_list = df_array.tolist()
te = TransactionEncoder()
te_ary = te.fit_transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.48, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules)

# 利用 fpgrowth_py完成关联规则分析，创建 FP树 ，并通过 FP树挖掘频繁项集 。
# frequent_itemsets2, fp_tree = fpgrowth(df_list,minSupRatio=0.5,minConf=0.5)

#6.2 特征分析
#构造绘图数据集
new_combined_data_plt = train_df_org.append(test_df_org,ignore_index=True)

#登船港口、舱室和船票价格的关系
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=new_combined_data_plt)
plt.axhline(y=80, color='r', linestyle='--')
plt.tight_layout()
plt.show()

#登船港口、舱室和年龄的关系
sns.boxplot(x="Embarked", y="Age", hue="Pclass", data=new_combined_data_plt)
plt.axhline(y=80, color='r', linestyle='--')
plt.tight_layout()
plt.show()

#性别、年龄和舱室之间的关系
sns.boxplot(x="Sex", y="Age",hue="Pclass", data=new_combined_data_plt)
plt.axhline(y=80, color='r', linestyle='--')
plt.tight_layout()
plt.show()

#------------------------------------------------------------------
#7.预测和进行模型评价
#------------------------------------------------------------------

#保留年龄、性别、客舱等级、登船港口、头衔、船票价格、家庭人数等特征
final_combined_data = combined_train_test
final_combined_data = final_combined_data.loc[:890,
            ['Survived','Age','Sex','Pclass'
            ,'Embarked', 'Title', 'Fare', 'Family_Size']]



# 从数据集中提取特征列
features = final_combined_data.drop('Survived', axis=1)
# 从数据集中提取目标列
target = final_combined_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# 建立决策树模型
# 记录开始时间
start_time = time.time()
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
DecisionTree_train_time = end_time - start_time
accuracy1 = accuracy_score(y_test, dt_model.predict(X_test))
print("DecisionTree Accuracy: ", accuracy1)
f1 = f1_score(y_test, dt_model.predict(X_test))
print("DecisionTree F1 Score: ", f1)
matrix = confusion_matrix(y_test, dt_model.predict(X_test))
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'DecisionTree Accuracy {accuracy1:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'DecisionTree ROC Curve AUC Score:{auc_score:.3%}')
plt.show()

# 建立朴素贝叶斯模型
# 记录开始时间
start_time = time.time()
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
MultinomialNB_train_time = end_time - start_time
# 预测
y_pred = nb_model.predict(X_test)
# 计算准确性
accuracy2 = accuracy_score(y_test, y_pred)
print("MultinomialNB Accuracy: ", accuracy2)
f1 = f1_score(y_test, y_pred)
print("MultinomialNB F1 Score: ", f1)
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'MultinomialNB Accuracy {accuracy2:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, nb_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'MultinomialNB ROC Curve AUC Score:{auc_score:.3%}')
plt.show()

# 建立神经网络模型
# 记录开始时间
start_time = time.time()
nn_model = MLPClassifier()
nn_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
MLPClassifier_train_time = end_time - start_time
# 预测
y_pred = nn_model.predict(X_test)
# 计算准确性
accuracy3 = accuracy_score(y_test, y_pred)
print("MLPClassifier Accuracy: ", accuracy3)
f1 = f1_score(y_test, y_pred)
print("MLPClassifier F1 Score: ", f1)
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'MLPClassifier Accuracy {accuracy3:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, nn_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'MLPClassifier ROC Curve AUC Score:{auc_score:.3%}')
plt.show()

# 建立支持向量机模型
# 记录开始时间
start_time = time.time()
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
SVC_train_time = end_time - start_time
# 预测
y_pred = svm_model.predict(X_test)
# 计算准确性
accuracy4 = accuracy_score(y_test, y_pred)
print("SVC Accuracy: ", accuracy4)
f1 = f1_score(y_test, y_pred)
print("SVC F1 Score: ", f1)
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'SVC Accuracy {accuracy4:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'SVC ROC Curve AUC Score:{auc_score:.3%}')
plt.show()

# 建立随机森林模型
# 记录开始时间
start_time = time.time()
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
RandomForest_train_time = end_time - start_time
# 预测
y_pred = rf_model.predict(X_test)
# 计算准确性
accuracy5 = accuracy_score(y_test, y_pred)
print("RandomForest Accuracy: ", accuracy5)
f1 = f1_score(y_test, y_pred)
print("RandomForest F1 Score: ", f1)
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'RandomForest Accuracy {accuracy5:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'RandomForest ROC Curve AUC Score:{auc_score:.3%}')
plt.show()

# 建立 K 近邻模型
# 记录开始时间
start_time = time.time()
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
# 记录结束时间
end_time = time.time()
# 计算训练时间
KNeighbors_train_time = end_time - start_time
# 预测
y_pred = knn_model.predict(X_test)
# 计算准确性
accuracy6 = accuracy_score(y_test, y_pred)
print("KNeighbors Accuracy: ", accuracy6)
f1 = f1_score(y_test, y_pred)
print("KNeighbors F1 Score: ", f1)
matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(matrix, cmap='Blues', annot=True, annot_kws={'size': 10})
plt.title(f'KNeighbors Accuracy {accuracy6:.3%}')
plt.show()
# 获得 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, knn_model.predict_proba(X_test)[:, 1])
auc_score = roc_auc_score(y_test, knn_model.predict_proba(X_test)[:, 1])
# 绘制 ROC 曲线
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'KNeighbors ROC Curve AUC Score:{auc_score:.3%}')
plt.show()


model_time = [DecisionTree_train_time
    , MultinomialNB_train_time, MLPClassifier_train_time
    , SVC_train_time, RandomForest_train_time,KNeighbors_train_time]
model_num = ['DecisionTree', 'MultinomialNB'
    , 'MLPClassifier', 'SVC', 'RandomForest', 'KNeighbors']
plt.plot(model_num, model_time)
# 设置 x 轴和 y 轴的标签
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.xticks(rotation=60)
# 显示图形
plt.tight_layout()
plt.show()

model_accuracy = [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6]
plt.plot(model_num, model_accuracy)
# 设置 x 轴和 y 轴的标签
plt.xlabel('Model')
plt.ylabel('Training Accuracy')
plt.xticks(rotation=60)
# 显示图形
plt.tight_layout()
plt.show()
