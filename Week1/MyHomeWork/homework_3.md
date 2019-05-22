# 对数几率回归(逻辑回归)算法的语言描述
**输入：** 数据集$D=\{(\boldsymbol{x}_1,y_1),(\boldsymbol{x}_2,y_2),...,(\boldsymbol{x}_m,y_m)\}, 其中\boldsymbol{x}_i=(x_{i1};x_{i2};...;x_{id}),y_i \in \{1, 0\}$  
**过程：**    
1. 初始化模型参数：$w\in R^n, b \in R$;
2. 建立对数几率回归模型：
$$\begin{array}{l}{p(y=1 | x)=\frac{e^{w^{T} x+b}}{1+e^{w^{T} x+b}}} \\ {p(y=0 | x)=\frac{1}{1+e^{w^{T} x+b}}}\end{array}$$
3. 令$\beta=(w;b), \hat{\boldsymbol{x}}=(\boldsymbol{x};1)$，则$$\begin{array}{l}{p_1(\hat {\boldsymbol{x}};\beta)=p(y=1 | \hat {\boldsymbol{x}};\beta)=\frac{e^{\beta^{T} x}}{1+e^{\beta^{T}x}}} \\ {p_0(\hat {\boldsymbol{x}};\beta)=p(y=1 | \hat {\boldsymbol{x}};\beta)=\frac{1}{1+e^{\beta^{T}x}}}\end{array}$$  
4. 计算式(3.27)关于$\beta$的高阶可导连续凸函数：$\ell(\beta)=\sum_{i=1}^{m}(-y_{i} \beta^{T} \hat{\boldsymbol{x}}_{i}+\ln (1+e^{\beta^{T} \hat{\boldsymbol{x}}_{i}}))$
5. 采用数值优化算法如梯度下降法、牛顿法求得最优解：$$\beta^*=\underset{\beta}{\arg \min } \ell(\beta)$$
6. 计算$w^*,b^*$，其中$(w^*,b^*)=\beta^*$
**输出：** $$\begin{array}{l}{p(y=1 | x)=\frac{e^{{w^*}^{T} x+b^*}}{1+e^{{w^*}^{T} x+b^*}}} \\ {p(y=0 | x)=\frac{1}{1+e^{{w^*}^{T} x+b^*}}}\end{array}$$