import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
tfb=tfp.bijectors
import utils as utl

# GMM基空间(Z)与均匀分布(U)之间的双射
# 沿每个维度的（联合GMM的）边缘CDF
class GMM_bijector(tfb.Bijector):
    def __init__(self, n_dims, n_comps, param_list,forward_min_event_ndims=1, validate_args: bool = False,name="gmm"):
        super(GMM_bijector, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        assert (len(param_list)==3), 'incorrect parameters'
        assert param_list[0].shape == n_comps,  'the dimension of weight vector shold be ncomps'
        assert param_list[1].shape == [n_comps,n_dims], 'the dimension of mean vectors should be ncomps x ndims'
        assert param_list[2].shape == [n_comps,n_dims], 'the dimension of variance vectors should be ncomps x ndims'
        
        self.ndims = n_dims
        self.ncomps = n_comps
        self.logits = param_list[0]
        self.mu_vectors = param_list[1]
        self.var_vectors = param_list[2]
        self.std_vectors = self.var_vectors**0.5
    # 前向变换
    def _forward(self, z_mat):
        assert z_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=self.logits),
                               components_distribution=tfd.Normal(loc=tf.transpose(self.mu_vectors),
                                                                  scale=tf.transpose(self.std_vectors)))
        u_mat = dist.cdf(z_mat)
        return u_mat
    
    # 反向变换
    def _inverse(self, u_mat):
        assert u_mat.shape[1] == self.ndims, 'expected data dimensions n_samps x n_dims'
        z_mat = self.gmm_icdf_parallel(u_mat, self.logits, tf.transpose(self.mu_vectors),tf.transpose(self.std_vectors))
        return z_mat
    
    # 计算反向变换的 Jacobian行列式
    # 逆变换的Jacobian行列式的对数等于正变换的Jacobian行列式的对数的相反数。
    def _inverse_log_det_jacobian(self, u_mat):
        z_mat = self._inverse(u_mat)
        dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=self.logits),
                               components_distribution=tfd.Normal(loc=tf.transpose(self.mu_vectors),
                                                                  scale=tf.transpose(self.std_vectors)))
        log_det_J_mat = dist.log_prob(z_mat)
        return -tf.reduce_sum(log_det_J_mat,axis=1)    
    
    # 沿着每个维度，并行地数值求解gmm边缘分布的icdf值
    @tf.custom_gradient
    def gmm_icdf_parallel(self,u_mat,logit,mu_T,std_T):
        # 计算逆CDF值可以转换为求u-ψ(z)=0的根
        # 定义函数u-ψ(z)
        obj_func = lambda z: tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logit),
                                                   components_distribution=tfd.Normal(loc=mu_T,scale=std_T)).cdf(z)-u_mat
        
        # 基于当前参数指定根的下限和上限
        lb = tf.reduce_min(mu_T,axis=1) - 5*tf.reduce_max(std_T,axis=1)
        ub = tf.reduce_max(mu_T,axis=1) + 5*tf.reduce_max(std_T,axis=1)
        lb = tf.repeat(tf.reshape(lb,[1,-1]),u_mat.shape[0],axis=0)
        ub = tf.repeat(tf.reshape(ub,[1,-1]),u_mat.shape[0],axis=0)
        
        # Chandrupatla算法求根
        z_mat = tfp.math.find_root_chandrupatla(obj_func,low=lb,high=ub)[0]
        
        # 自定义梯度，根据 dy 计算并返回梯度
        def grad(dy):
            # 调用partial_deriv_z计算gmm边缘icdf的解析偏导数
            grad_logit, grad_mu, grad_std = self.partial_deriv_z(z_mat,logit,mu_T,std_T)

            temp_mat = tf.linalg.matmul(grad_logit,dy)

            logit_grad = tf.linalg.diag_part(temp_mat)
            logit_grad = tf.reduce_sum(logit_grad,axis=1)

            temp_mat = tf.linalg.matmul(grad_mu,dy)
            mu_grad = tf.linalg.diag_part(temp_mat)

            temp_mat = tf.linalg.matmul(grad_std,dy)
            std_grad = tf.linalg.diag_part(temp_mat)
            
            return tf.constant(0.,shape=(u_mat.shape)), logit_grad, tf.transpose(mu_grad), tf.transpose(std_grad)    
        return z_mat, grad
    
    # gmm边缘icdf的解析偏导数
    def partial_deriv_z(self,z,logit,mu_T,std_T):
        alpha = tf.math.softmax(logit)
        grad_logit_array = tf.TensorArray(tf.float32, size=self.ncomps)
        grad_mu_array = tf.TensorArray(tf.float32, size=self.ncomps)
        grad_var_array = tf.TensorArray(tf.float32, size=self.ncomps)        
        dist = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(logits=logit),
                                components_distribution=tfd.Normal(loc=mu_T,
                                                                  scale=std_T))
        
        common_factor1 = dist.prob(z) # 分母中的表达式对于所有偏导数都是相同的，并且只是单变量 GMM 的密度函数
        for k in range(self.ncomps):        
            common_factor2 = tfd.Normal(loc=mu_T[:,k],scale=std_T[:,k]).prob(z)
            term = 0.5*(1+tf.math.erf((z-mu_T[:,k])/(tf.math.sqrt(2.)*std_T[:,k])))
            v1 = -alpha[k]*(term - dist.cdf(z))/common_factor1
            v2 = alpha[k]*common_factor2/common_factor1
            v3 = v2 * ((z-mu_T[:,k])/(std_T[:,k])) 

            grad_logit_array = grad_logit_array.write(k, tf.transpose(v1) )
            grad_mu_array = grad_mu_array.write(k, tf.transpose(v2) )
            grad_var_array = grad_var_array.write(k, tf.transpose(v3) )
        return grad_logit_array.stack(), grad_mu_array.stack(), grad_var_array.stack()
    
# 均匀分布(U)与真实数据(X)之间的双射
# 每个维度上先验学习的边缘CDF的逆函数
class Marginal_transform(tfb.Bijector):
    def __init__(self,ndims,marg_dist_list,forward_min_event_ndims=1, validate_args: bool = False,name="marginals"):
        super(Marginal_transform, self).__init__(
            validate_args=validate_args, forward_min_event_ndims=forward_min_event_ndims, name=name
        )
        self.ndims = ndims
        self.marg_dists = marg_dist_list    # 单独学习的真实数据的边缘密度 
    
    # 前向变换 
    def _forward(self, u_mat):
        temp_array = tf.TensorArray(tf.float32,size=self.ndims)
        for j in range(self.ndims):
            if 'icdf' in self.marg_dists[j]:
                x_cur = self.marg_dists[j]['icdf'](u_mat[:,j])
            # 如果边缘分布没有提供逆CDF，那么会使用数值方法（用求根方法替换）来计算
            else:
                x_cur = utl.icdf_numerical(u_mat[:,j], 
                                       self.marg_dists[j]['cdf'],
                                       self.marg_dists[j]['lb'],
                                       self.marg_dists[j]['ub'])
            temp_array = temp_array.write(j,x_cur)
        x_mat = tf.transpose(temp_array.stack())              
        return x_mat
    
    # 反向变换
    def _inverse(self, x_mat):
        temp_array = tf.TensorArray(tf.float32,size=self.ndims)
        for j in range(self.ndims):
            u_cur = self.marg_dists[j]['cdf'](x_mat[:,j])
            temp_array = temp_array.write(j,u_cur)
        u_mat = tf.transpose(temp_array.stack())            
        return u_mat
    
    # 计算正向变换的Jacobian行列式
    def _forward_log_det_jacobian(self, u_mat):
        x_mat = self._forward(u_mat)
        temp_array = tf.TensorArray(tf.float32,size=self.ndims)
        for j in range(self.ndims):
            temp_array = temp_array.write(j,self.marg_dists[j]['log_pdf'](x_mat[:,j]))
        log_det_J_mat = tf.transpose(temp_array.stack())
        return -tf.reduce_sum(log_det_J_mat,axis=1) 
    
    # 计算反向变换的Jacobian行列式
    def _inverse_log_det_jacobian(self, x_mat):
        u_mat = self._inverse(x_mat)
        return -self._forward_log_det_jacobian(u_mat)
