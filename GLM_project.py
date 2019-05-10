#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#----------------------------------------数据载入----------------------------------------------------------------
#训练数据
train_data = pd.read_csv(r"D:\Code\Jupyter-notebook\data\2001_SPECTF Heart Data Set\SPECTF_train.txt",header=None)
#测试数据
test_data = pd.read_csv(r"D:\Code\Jupyter-notebook\data\2001_SPECTF Heart Data Set\SPECTF_test.txt",header=None)

#添加特征名称行
train_data.columns = ['OVERALL_DIAGNOSIS','F1R','F1S','F2R','F2S','F3R','F3S','F4R','F4S','F5R','F5S','F6R','F6S','F7R','F7S','F8R','F8S',
                     'F9R','F9S','F10R','F10S','F11R','F11S','F12R','F12S','F13R','F13S','F14R','F14S','F15R','F15S',
                      'F16R','F16S','F17R','F17S','F18R','F18S','F19R','F19S','F20R','F20S','F21R','F21S','F22R','F22S']
test_data.columns = ['OVERALL_DIAGNOSIS','F1R','F1S','F2R','F2S','F3R','F3S','F4R','F4S','F5R','F5S','F6R','F6S','F7R','F7S','F8R','F8S',
                     'F9R','F9S','F10R','F10S','F11R','F11S','F12R','F12S','F13R','F13S','F14R','F14S','F15R','F15S',
                      'F16R','F16S','F17R','F17S','F18R','F18S','F19R','F19S','F20R','F20S','F21R','F21S','F22R','F22S']

print(train_data.shape,test_data.shape)
train_data.head()



# In[2]:


#----------------------------------------数据信息--------------------------------------------------------
'''训练数据集比较简单，均为连续数值，没有缺失值，数据大小在同一维度。百分之50诊断为1。'''
print(train_data.info())
# train_data.isnull().sum()   
train_data.describe()


# In[3]:


'''测试数据集比较简单，均为连续数值，没有缺失值，数据大小在同一维度。百分之90诊断为1'''
print(test_data.info())
# test_data.isnull().sum()
test_data.describe()


# In[4]:


#训练集和测试集的target数量
plt.subplot(121)
train_data['OVERALL_DIAGNOSIS'].value_counts().plot.pie(autopct='%.2f%%',startangle=90)
plt.title("spectf_train")
plt.subplot(122)
test_data['OVERALL_DIAGNOSIS'].value_counts().plot.pie(autopct='%.2f%%',startangle=90)
plt.title("spectf_test")
'''测试集的分布不均匀，1明显多过0'''


# In[5]:


#看看训练集和测试集的数据分布是否一致

for col in train_data.columns:
    sns.distplot(train_data[col])
    sns.distplot(test_data[col])
    plt.show()
'''可以看出训练集和测试集的数据分布基本一致，差异不算太过显著。'''


# In[6]:


#----------------------------------------建模------------------------------------------------------------------
#模型1.full model使用全部特征+L1正则项
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split


X_train = train_data.drop('OVERALL_DIAGNOSIS',axis=1)
y_train = train_data['OVERALL_DIAGNOSIS']
X_test = test_data.drop('OVERALL_DIAGNOSIS',axis=1)
y_test = test_data['OVERALL_DIAGNOSIS']

#划分训练和验证集
# X_train,X_validation,y_train,y_validation = train_test_split(train_data,train_target,test_size=0.25,random_state=0)

LR = LogisticRegression(penalty='l1')
LR.fit(X_train,y_train)
print('模型的截距：{}\n协变量权重{}'.format(LR.intercept_,LR.coef_))
# print('-'*80)
# score = LR.score(X_test,y_test)
# print('模型得分:',score)

#饱和模型中，'F2S','F4R','F8S','F9R','F10R','F11R','F11S','F17R','F19S','F22R'的权重为0


# In[7]:


#####################以下做模型优化


# In[8]:


###模型2.使用PCA降维+L1正则项
from sklearn.decomposition import PCA
#mle方法自动选择方差适合的特征
pca = PCA(n_components=20,copy=True)
X_train_reduced = pca.fit_transform(X_train)
print("降维后特征数：{}，他们方差占比为：{}" .format(pca.n_components_,pca.explained_variance_ratio_.sum()))
# pca.explained_variance_
# pca.components_
X_train_reduced.shape
'''
降维后特征数：14，他们方差占比为：0.9010059067639149
降维后特征数：20，他们方差占比为：0.9498533198434759
降维后特征数：32，他们方差占比为：0.9908832820253627

'''


LR2 = LogisticRegression(penalty='l1')
LR2.fit(X_train_reduced,y_train)

X_test_reduced = pca.transform(X_test)
print("模型的参数设置：",LR2.get_params)
print('-'*80)
print('模型的截距：{}\n协变量权重{}'.format(LR2.intercept_,LR2.coef_))
# print('-'*80)
# score2 = LR2.score(X_test_reduced,y_test)
# print('模型得分:',score2)
'''
a.若选择降维后特征数为14，penalty为L1，LR2得分为 0.6844919786096256
b.若选择降维后特征数为14，penalty为L2，LR2得分为 0.6844919786096256
c.若选择降维后特征数为20，penalty为L1，LR2得分为 0.6951871657754011(*)
d.若选择降维后特征数为20，penalty为L2，LR2得分为 0.6898395721925134
e.若选择降维后特征数为32，penalty为L1，LR2得分为 0.6951871657754011(*)
f.若选择降维后特征数为32，penalty为L2，LR2得分为 0.6470588235294118

结论：因为32维度仍然过大，所以选择20维，得分仍然最高，虽然损失了5%的信息
'''


'''[结论1]：从上面分析得知，c(降维后特征数为20，penalty为L1)的泛化能力最强，虽然丢失了5%的原始数据信息。
由pca选出的20个特征训练模型，从结果可以看出有些特征的权重很小,甚至为零。
'''


# In[9]:


#模型3 基于AIC的最优模型
#查看相关系数
train_corr = train_data.corr(method='pearson')
train_corr['OVERALL_DIAGNOSIS'].sort_values(ascending=False)
#从各自变量和因变量的相关系数看，只有F20S和OVERALL_DIAGNOSIS中度相关(abs>0.5)

#使用向前逐步法从其它备选变量中选择变量
import statsmodels.formula.api as smf
import statsmodels.api as sm
data = pd.concat([X_train,y_train],axis=1)


# 向前逐步法
def forward_select_aic(data,response):
    remaining=set(data.columns)
    remaining.remove(response)
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')
    while remaining:
        aic_with_candidates=[]
        for candidate in remaining:
            formula='{}~{}'.format(
                response,'+'.join(selected+[candidate]))
            aic=smf.glm(
                formula=formula,data=data,
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().aic
            aic_with_candidates.append((aic,candidate))
        aic_with_candidates.sort(reverse=True)
        best_new_score,best_candidate=aic_with_candidates.pop()
        if current_score>best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score=best_new_score
            print('aic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break
    formula='{}~{}'.format(response,'+'.join(selected))
    print('final formula is {}'.format(formula))
    model=smf.glm(
        formula=formula,data=data,
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)

LR3 = forward_select_aic(data=data,response='OVERALL_DIAGNOSIS')
print(LR3.summary())

#有的系数没通过p值检验，如何优化？


# In[10]:


#模型4 基于BIC的最优模型
# 向前逐步法
def forward_select_bic(data,response):
    remaining=set(data.columns)
    remaining.remove(response)
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')
    while remaining:
        bic_with_candidates=[]
        for candidate in remaining:
            formula='{}~{}'.format(
                response,'+'.join(selected+[candidate]))
            bic=smf.glm(
                formula=formula,data=data,
                family=sm.families.Binomial(sm.families.links.logit)
            ).fit().bic
            bic_with_candidates.append((bic,candidate))
        bic_with_candidates.sort(reverse=True)
        best_new_score,best_candidate=bic_with_candidates.pop()
        if current_score>best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score=best_new_score
            print('bic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break
    formula='{}~{}'.format(response,'+'.join(selected))
    print('final formula is {}'.format(formula))
    model=smf.glm(
        formula=formula,data=data,
        family=sm.families.Binomial(sm.families.links.logit)
    ).fit()
    return(model)

LR4 = forward_select_bic(data=data,response='OVERALL_DIAGNOSIS')
print(LR4.summary())


# In[11]:


#----------------------------------------------模型评估-----------------------------------------------------------
from sklearn import metrics

# LR模型预测
test_data['LR_pred']=LR.predict(X_test)

# LR2模型预测
test_data['LR2_pred']=LR2.predict(X_test_reduced)

# LR3模型预测
test_data['LR3_prob'] = LR3.predict(X_test)


# LR4模型预测
test_data['LR4_prob'] = LR4.predict(X_test)


# 计算准确率
LR3_pred=(test_data['LR3_prob']>0.5).astype('int')
LR4_pred=(test_data['LR4_prob']>0.5).astype('int')
test_data['LR3_pred'] = LR3_pred
test_data['LR4_pred'] = LR4_pred

LR_acc=sum(test_data['LR_pred']==y_test)/np.float(len(y_test))
LR2_acc=sum(test_data['LR2_pred']==y_test)/np.float(len(y_test))
LR3_acc=sum(test_data['LR3_pred']==y_test)/np.float(len(y_test))
LR4_acc=sum(test_data['LR4_pred']==y_test)/np.float(len(y_test))

print('The accurancy of LR is %.2f'%LR_acc,'\n')
print('The accurancy of LR2 is %.2f'%LR2_acc,'\n')
print('The accurancy of LR3 is %.2f'%LR3_acc,'\n')
print('The accurancy of LR4 is %.2f'%LR4_acc,'\n')

# 混淆矩阵   
LR_confusion_matrix=pd.crosstab(test_data.LR_pred,y_test,margins=True)
LR2_confusion_matrix=pd.crosstab(test_data.LR2_pred,y_test,margins=True)
LR3_confusion_matrix=pd.crosstab(test_data.LR3_pred,y_test,margins=True)
LR4_confusion_matrix=pd.crosstab(test_data.LR4_pred,y_test,margins=True)

 # 计算评估指标    
print('LR评估指标','\n',metrics.classification_report(y_test, test_data.LR_pred)) 
print('LR2评估指标','\n',metrics.classification_report(y_test, test_data.LR2_pred))  
print('LR3评估指标','\n',metrics.classification_report(y_test, test_data.LR3_pred)) 
print('LR4评估指标','\n',metrics.classification_report(y_test, test_data.LR4_pred)) 

# 绘制Roc曲线    
import sklearn.metrics as metrics


LR_fpr_test,LR_tpr_test,LR_th_test=metrics.roc_curve(y_test,LR.predict_proba(X_test)[:,1])#转为prob

LR2_fpr_test,LR2_tpr_test,LR2_th_test=metrics.roc_curve(y_test,LR2.predict_proba(X_test_reduced)[:,1])

LR3_fpr_test,LR3_tpr_test,LR3_th_test=metrics.roc_curve(y_test,test_data.LR3_prob)

LR4_fpr_test,LR4_tpr_test,LR4_th_test=metrics.roc_curve(y_test,test_data.LR4_prob)



plt.plot(LR_fpr_test,LR_tpr_test,'b--',label='LR')
plt.plot(LR2_fpr_test,LR2_tpr_test,'y--',label='LR2')
plt.plot(LR3_fpr_test,LR3_tpr_test,'g--',label='LR3')
plt.plot(LR4_fpr_test,LR4_tpr_test,'r--',label='LR4')
plt.title('ROC_test curve')
plt.legend(loc='best')
plt.show()

print('LR_AUC=%.4f'%metrics.auc(LR_fpr_test,LR_tpr_test))
print('LR2_AUC=%.4f'%metrics.auc(LR2_fpr_test,LR2_tpr_test))
print('LR3_AUC=%.4f'%metrics.auc(LR3_fpr_test,LR3_tpr_test))
print('LR4_AUC=%.4f'%metrics.auc(LR4_fpr_test,LR4_tpr_test))

'''
【结论】
1.从四个模型的预测准确率看，饱和模型的准确率最低，为0.61，其余三个模型准确率都达到70%以上；
2.从四个模型的召回率看，饱和模型对0的召回率低于50%，其余三者高于50%，且基于BIC的模型具有最高召回率；
3.从四个模型的F1得分看，LR<LR2<LR3<LR4.
4.从四个模型的测试数据的ROC曲线和AUC值看，LR4模型优于其他模型。
综上，模型四是最佳模型。
'''

