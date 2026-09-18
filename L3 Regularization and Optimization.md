# Regularization
Last lecture, we use SVM and Softmax to find weights W whose predictions match the labels in the training data best. However, there are left questions:
1. Is the best W unique? 
	Suppose we use multiclass SVM loss, it happens when W, $L_i$ = 0, and when 2W, $L_i$ = 0. Same for 3W, 4W, ... 
	Which one is better? 
2. Will it be overfitting the training set?(More important)
	Training performance $\neq$ generalization performance. 
	How can we avoid model to obtain excellent training performance by learning noise and irrelevant details?
---
Our principle for regularization: Among competing explanations that fit the evidence, prefer the simpler one. (Occam's razor)
So for model, we prefer the one relying on less extreme or less complicated assumptions.

---
The complete objective function: 
$$
L(W) = L_{\text{data}}(W)+\lambda R(W)
$$
data-loss term is in charge with fitting the training labels:
$$
L_{\text{data}}(W)=\frac{1}{N}\sum_{i=1}^{N}L_i.
$$
regularization term is for choosing parameter values with desirable properties:
including: small weights, sparse weights, smooth predictions, robustness to perturbations, reduced dependence on individual units
The regularization strength $\lambda$(hyperparameter): 
	when small: fitting the data remain dominant -> risk of overfitting
	when large: care more about keeping the weight simple -> risk of underfitting
	
---
Common choice: L2 regularization
$$
R(W) = \sum_kW_k^2 = ||W||^2_F\text{ (for matrix)}
$$
so for the question we raised at the beginning, L(W) = 0 + $\lambda||W||_F^2$ , while L(2W) is 4 times of this.
Also, L2 regularization likes to spread out the weights: e.g. 
$$
x = \begin{bmatrix}
1 \\
1 \\
1 \\
1
\end{bmatrix}
\qquad
w_1 = \begin{bmatrix}
1 \\
0 \\
0 \\
0
\end{bmatrix}
\qquad
w_2 = \begin{bmatrix}
0.25 \\
0.25 \\
0.25 \\
0.25
\end{bmatrix}
$$
so both weights produce $w^Tx = 1$, but $||w_1||_2^2 = 1$, $||w_2||_2^2 = 4(0.25)^2 = 0.25$
In image classification, this imply that the classifier is encouraged to collect evidence from multiple pixels.
Another choice: L1 regularization
$$
R(W) = \sum_k|W_k|
$$
Generally, L1 regularization tends to produce sparse weights(many components are exactly 0)
Elastic-net regularization: we combine L1 and L2
$$
R(W) = \alpha||W||_1 + \beta||W||_2
$$
it tries to gain both sparsity from L1 and smooth, distributed weights from L2.

---
Advantages of Regularization:
1. express preference over weights
2. improve performance on unseen data
3. improve optimization-> it can change geometric shape of the loss function
# Optimization
First part demonstrate what we want, and optimization is for how to get it.
Out optimization problem:
$$
W^* = \arg\min_WL(W)
$$
First strategy we consider may be random guess, but it is hopelessly inefficient. Also, we want something that can improve little by little with a direction. Thus it comes to gradient.
For loss $L(W_1,W_2,...,W_D)$, 
$$ \nabla_{\mathbf{W}} L = \begin{bmatrix} \frac{\partial L}{\partial W_1} \\ \frac{\partial L}{\partial W_2} \\ \vdots \\ \frac{\partial L}{\partial W_D} \end{bmatrix} $$
we want to move towards the direction L decrease, i.e. $-\nabla_{\mathbf{W}} L$, we call this direction steepest descent. Update rule:
$$
\mathbf{W} \leftarrow \mathbf{W} - \eta\nabla_{\mathbf{W}} L, 
\qquad
\eta \text{ is the learning rate, controlling how far we step}
$$
To calculate the gradient, consider numerical gradient first: 
$$
\frac{\partial L}{\partial W_k}
\approx
\frac{L(\mathbf{W} + h\mathbf{e}_k) - L(\mathbf{W})}{h},
\qquad
e_k \text{ is 0 everywhere except position k where it is 1, h is very small number}
$$
($e_k$ for selecting, h for derivative)
But since D may be very large for so many calculations, and h is hard to choose, so we don't use numerical gradients for training since they are slow and approximate. But we can use it to check our derivative code.
Consider analytic gradient instead: using chain rule
Also for the scale: If we choose full batch, we will need to calculate gradient descent for every training example, and this is expensive for a huge dataset to do this each time we update. 
Instead, we choose a small random subset, which we call a minibatch. This method is usually called stochastic gradient descent, or SGD, althoght it should be minibatch SGD for formal name.
If the minibatch is B, then the gradient now would be: 
$$
\nabla_{\mathbf{W}} L_B(\mathbf{W})
=
\frac{1}{|B|}
\sum_{i \in B}
\nabla_{\mathbf{W}} L_i
$$
update by: $$
\mathbf{W}
\leftarrow
\mathbf{W}
-
\eta \nabla_{\mathbf{W}} L_B(\mathbf{W})
$$
But SGD has its problems.
1. poor conditioning
	Some directions have much stronger curvature than others, which makes one fixed learning rate inefficient.
	e.g. A valley-like shape. Across the valley, the loss rises sharply. Along the valley, it changes slowly. But gradient descent tends to point strongly across the steep direction, so it may dangling aross the valley over and over again(z 字型). Fixed learning rate is also not working, since we need different rate for different directions.
2. local minima and saddle points
	This happen cuz we rely on gradient. It happens that the point is not the general minimum, but is a local minimum, and the model would stuck there. 
3. noisy minibatch gradients
	Minibatch is just a sample of the full dataset, so if the minibatch is not representative enough, it might create additional noise. Sometimes it can help the optimizer escape saddle points or flat regions, but sometimes it makes the path unstable and slow convergence.

## Momentum(An optimization method)
For plain minibatch SGD: 
$$
\mathbf{W}_{t+1} = \mathbf{W}_t - \eta\nabla L_B(\mathbf{W}_t)
$$
the update has no memory: we compute gradient completly from current W. So we have no idea of the path we walked. Thus we introduce momentum to add memory of previous step.
We use velocity $v_t$ to represent direction and strength parameters have been moving recently. 
A common update is:
$$
v_t = \underbrace{\mu v_{t-1}}_{\text{remember old movement}} - \underbrace{\eta \nabla L(W_{t-1})}_{\text{use new gradient}},
\qquad
W_t = W_{t-1} + v_t
$$
so since $v_t$ is a weighted sum of recent gradient, the problem we mentioned above, about moving directions frequently, the opposite direction would cancel each other while accumulating to be $v_t$, thus to keep the moving direction stably ahead.
Also, accumulated velocity can keep the optimization moving even in flat regions. 
You could see the first term of $v_t$ as inertia（惯性）, and the second term to be new direction. So $\mu$(usually)$\in[0,1)$ denotes how strong the memory is. 

 --- 
 What we discussed above is the standard momentum, calculate the gradient at current location, then moves using its velocity. But we can also introduce expected gradient.
 Nesterov momentum is written by: 
 $$
v_t = \mu v_{t-1} - \eta \nabla L(W_{t-1} + \mu v_{t-1}),
\qquad


W_t = W_{t-1} + v_t
$$
gradient:
$$
\nabla L (W_{t-1} + \mu v_{t-1}) 
$$
This allow us to take our expected next move into consideration, and correct the velocity before we take this move. 
## Adaptive learning rates($\eta$): AdaGrad, RMSProp, Adam
 We want the parameter can automatically get the suitable learning rate, instead of setting a fixed one for it, which is inflexible.
 We have several choices for this:
 1. AdaGrad
	 Let $g_t$ denote the gradient at step t: $g_t = \nabla L(W_t)$
	 AdaGrad stores $G_t = G_{t-1}+g_t^2$, and update is:
	 $$
W_{t+1}
=
W_t
-
\frac{\eta}{\sqrt{G_t} + \epsilon} g_t,
\qquad
\epsilon \text{ is a tiny number to prevent division by 0}
$$
	The rule is: if you have take lots of changes, $G_t$ would be big, thus new $\eta'$ would become small, taking smaller step; if you have barely changed, then $\eta'$ would be big. This is especially useful for sparese features: it can take rare data effectively.
	However, $G_t$ monotonically increasing, thus $\eta'$ monotonically decreasing, it would eventually become very small and the training process would stuck.
2. RMSProp
	It fix the problems above by using a moving average instead of full accumulated sum
	$$
	s_t = \rho s_{t-1} + (1-\rho) g_t^2,
	\qquad
	\rho \text{ is often around 0.9 or 0.99}
	$$
	update is:
	$$
	W_{t+1} = W_t - \frac{\eta}{\sqrt{s_t} + \epsilon} g_t,
	$$
	Since $\rho<1$, gradient from many steps ago would only have little influence now. 
3. Adam: Adaptive Moment Estimation, momentum + RMSProp
	Combine average the gradient(direction) and average squared gradient(step size).
	$m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
	update:
	$$
	W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t,
	$$
	```Python
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	```
	At the beginning of training, $m_0$ = 0, $v_0$ = 0, no training history, so automatically treat as 0. This cause a bias, since normally the past should close to the present gradient instead of 0. So here PyTorch would automatically do bias correction as:(suppose $\beta_1$ = 0.1)
	$$
	\hat{m_1} = \frac{m_1}{1-\beta_1} = g_1 
	\qquad
	\leftarrow
	\qquad
	m_1 = \beta_1(0) + (1-\beta_1)g_1 = 0.1g_1
	$$
	same for $v_1$. 
--- 
For these optimizer, they still contain $\eta$ . Since they only tackle the scale problems, and try to let $\eta'$ moving within specific scales. They adjust the scale per parameter, but the base $\eta$ still takes an important role. But we usually don't want a fixed learning rate: its like filling color into a square, first we can fill quick at the middle, then we need to fill slow and careful when close to the edge. 
	The simplest schedue is to decrease learning rate at selected epochs(**step decay**). e.g. 0.1 at the fisrt 30 epochs $\rightarrow$ 0.01 at the next 30 epochs $\rightarrow$ 0.001 after.
	We could also shirink the learning rate soomthly(**exponential decay**): $\eta_t = \eta_0\gamma^t$, $\gamma \in (0,1)$ as decay factor. 
	Also (and popular) we can do **cosine decay**: 
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) 
(1 + \cos(\frac{\pi t}{T}))
$$
	The shape is: starts near $\eta_{\max}$, decreases smoothly, and ends near $\eta_{\min}$.
```Python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
scheduler.step() # call during the training
```
However, modern models often begins with a tiny one, cuz the very first stage of training can be unstable since we are still setting everything and has little information. In order to tackle this, we use learning-rate warmup: starting small and increasing gradually. For $T_{\text{warmup}}$ steps, initial learning rates would be: $\eta_t = \eta_{\max}\frac{t}{T_{\text{warmup}}}$.

## Second-order optimization
For all the optimiation above, we use first-order optimization: using first derivative $\nabla L(W)$ of the loss. But this is not the only choice.
Second derivative study curvature. This tells us whether a big step will go opposite from the prediction of gradient(which is very local). 
Newton's method(derived from second-order Taylor expansion): 
$$
w_{t+1} = w_t - \frac{f'(w_t)}{f''(w_t)}
$$
This is like we replace $\eta$ in gradient descent by $\frac{1}{f''(w_t)}$. Thus in a steeply curved direction, we take smaller step, and in flatter direction, we take bigger step. 
For a parameter vector, we use Hessian matrix($D\times D$) for second derivative: 
$$
H =
\begin{bmatrix}
\frac{\partial^2 L}{\partial W_1^2}
&
\frac{\partial^2 L}{\partial W_1 \partial W_2}
&
\cdots
\\
\frac{\partial^2 L}{\partial W_2 \partial W_1}
&
\frac{\partial^2 L}{\partial W_2^2}
&
\cdots
\\
\vdots
&
\vdots
&
\ddots
\end{bmatrix}
$$
Newton's method becoms:
$$ 
W_{t+1} = W_t - H^{-1}\nabla L(W_t)
$$
This is a good idea, but again, computation is extremely expensive. Thus standard deep learning training mainly use first-order derivative. 