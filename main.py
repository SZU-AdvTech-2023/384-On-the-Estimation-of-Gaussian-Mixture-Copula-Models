import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy as sc
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from GMCM import GMCM
import utils as utl
from sklearn.datasets import load_iris, load_wine
plt.style.use('ggplot')

def main():

    data=load_iris().data.astype('float32')
    # data=load_wine().data.astype('float32')
    
    nsamps,ndims = data.shape
    # 定义预处理双射变换（对数变换）
    # 这个可选的双射变换可以链接到GMCM双射
    # 它通过减轻重尾来提高对边缘的学习
    min_val = (np.min(data)-3*np.std(data)).astype('float32')
    log_transform = tfb.Chain([tfb.Shift(shift=min_val),tfb.Exp()])
    print(f'Number of samples = {nsamps}, Number of dimensions = {ndims}')

    # 将数据划分为训练集、测试集和验证集
    data_trn,data_vld,data_tst = utl.splitData(data)
    
    # 使用scikit-learn GMM（具有相同数量的组件）
    gmm=utl.GMM_best_fit(data_trn,
                            min_ncomp=2,
                            max_ncomp=10,
                            max_iter=10000, 
                            print_info=True)
    print("number of components: ", gmm.n_components)
    
    # 初始化GMCM对象
    gmcm=GMCM(ndims, data_transform=log_transform)
    # gmcm=GMCM(ndims)

    # 训练GMCM
    nll_train,nll_vld,_=gmcm.fit_dist_likelihood(data_trn,
                                        n_comps=gmm.n_components,
                                        batch_size=10,
                                        max_iters=6001,
                                        regularize=True,
                                        init='random', 
                                        print_interval=1000)

    # 展示迭代过程
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(-nll_train, c='#1f77b4')
    ax.set_xlabel('Iterations',fontsize=16)
    ax.set_ylabel('Log-likelihood',fontsize=16)
    plt.show()
    
    # 沿着任意两个维度获取（GMCM和GMM的）边缘分布
    dim1,dim2=0,1

    # GMCM 边缘分布
    gmcm_marg=gmcm.get_marginal([dim1,dim2])

    # GMM 边缘分布
    gmm_marg_mus=gmm.means_[:,[dim1,dim2]].astype('float32')
    gmm_marg_covs=tf.gather(tf.gather(gmm.covariances_.astype('float32'),[dim1,dim2],axis=1),[dim1,dim2],axis=2)
    gmm_marg_alphas=gmm.weights_.astype('float32')
    gmm_marg=tfd.MixtureSameFamily(tfd.Categorical(probs=gmm_marg_alphas),
                                tfd.MultivariateNormalFullCovariance(loc=gmm_marg_mus,
                                                                        covariance_matrix=gmm_marg_covs))
    
    # 沿两个指定尺寸绘制密度等值线
    fig,ax=plt.subplots(1,2,figsize=(12,4))

    plt.sca(ax[0])
    utl.plotDensityContours(data,gmcm_marg.distribution.log_prob,dim1,dim2)
    ax[0].set_title('GMCM density contours',fontsize=16)

    plt.sca(ax[1])
    utl.plotDensityContours(data,gmm_marg.log_prob,dim1,dim2)
    ax[1].set_title('GMM density contours',fontsize=16)
    plt.show()
    
    # 展示聚类结果
    df = pd.DataFrame(data)
    # 添加具有预测标签的列
    df['gmcm_label']=gmcm.predict(data)
    df['gmm_label']=gmm.predict(data)

    # Pairwise Scatter plotting with marker colors indicating component labels
    fig,ax=plt.subplots(1,2,figsize=(12,4))

    # plt.sca(ax[0])
    sns.scatterplot(df,x=dim1,y=dim2,hue='gmcm_label',palette='Spectral',edgecolor='k',ax=ax[0])
    ax[0].set_title('Scatter Plot with GMCM predicted labels',fontsize=16)
    ax[0].set(xlabel='dim_0', ylabel='dim_1')

    # plt.sca(ax[1])
    sns.scatterplot(df,x=dim1,y=dim2,hue='gmm_label',palette='Spectral',edgecolor='k',ax=ax[1])
    ax[1].set_title('Scatter Plot with GMM predicted labels',fontsize=16)
    ax[1].set(xlabel='dim_0', ylabel='dim_1')
    plt.show()
    
main()
