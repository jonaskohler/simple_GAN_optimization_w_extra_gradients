# simple_GAN_optimization_w_extra_gradients
2D toy examples for minmax problems with extra gradient descent.

This notebook explores the gradient descent dynamics on a simple,two-dimensional "GAN" example. While the *true* dataset consists of scalars drawn uniformly within $[0,1]$, the *noise* data is randomly sampled from $U[1,2]$. Discriminator and Generator are simple linear models in the form of $Wx+b$, where we fix $W_G=W_D=1$ to keep the dimensionality low. The following non-linear activation function is applied by the discriminator:

d(x)=exp(-(W_Dx+b_D)^2)

As a result, the Generator must learn to shift its input by $b_G=-1$ in order to successfuly re-generate the true data. The setting is inspired by: http://www.araya.org/archives/1183


![Toy example](/loss_landscape_gradient_flow_and_iterates.png)
