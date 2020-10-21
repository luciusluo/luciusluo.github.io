---
layout: post
title:  "From Backprop to BPTT"
date:   2020-06-27 12:43:01 -0700
author: By Lucius Luo
categories: jekyll update
---

<!--
<style type='text/css'>
  h3{
    color: #2a7ae2;
  }
</style>
-->

<h3><a id="hint"></a>A Hint of History</h3>

The **Back-propagation** Algorithm is, with no doubt, one of the most important and powerful mathematical tool used by a variety of machine learning models. Using the chain rule and partial derivative, it computes the gradient of the cost function with respect to each element in the weight matrix at each layer of the neural network. And this calculation informs us how fast the cost will change when we change the weights and biases within our network. 

First proposed by [Seppo Linnainmaa](http://people.idsia.ch/~juergen/who-invented-backpropagation.html)
in 1970, back-propagtion was later introduced to train neural network in 1974 by [Paul Werbos](http://www.werbos.com/) in his famous PhD [dissertation](https://www.wiley.com/en-us/The+Roots+of+Backpropagation%3A+From+Ordered+Derivatives+to+Neural+Networks+and+Political+Forecasting+-p-9780471598978). But the algorithm did not gain enough appreciation until [a famous paper](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) in 1986 by [David Rumelhart](https://en.wikipedia.org/wiki/David_Rumelhart), [Geoffrey Hinton](https://www.cs.toronto.edu/~hinton/), and [Ronald Williams](https://en.wikipedia.org/wiki/Ronald_J._Williams), who achieved some breakthrough success in several supervised learning tasks. 

Rumelhart and et al.'s paper, illustrating how back-propagation can adjust the weights of the network to better minimize the error between actual and desired output vectors, demonstrates the algorithm's ability to create new features, work faster, and sovle problems that were "insoluble" by some earlier approaches, including the [Perceptrons](https://en.wikipedia.org/wiki/Perceptron).

Not long after Rumelhart and et al.'s paper came out, a series of variants of the Backprop model were also invented, among which the most important ones are the **Back-propagation Through Time (BPTT)**, **Epochwise BPTT**, **Truncated BPTT (TrBPTT)**, and **Real Time Recurrent Learning (RTRL)**, again all designed by Paul Webos in his another influential [paper](http://axon.cs.byu.edu/Dan/678/papers/Recurrent/Werbos.pdf). 

In this post, I will illustrate how the simple, initial Backprop model evolves into the later BPTT and its variants, which further lay down a solid foundation for another powerful model: [Long Short-Term Memory (LSTM)](https://en.wikipedia.org/wiki/Long_short-term_memory). I will talk about LSTM in the next post. 
 

- <a href="#hint">A History of History</a>
- <a href="#bp">How Does Backpropagation Work?</a>
- <a href="#train_bp">How to train in Backpropagation?</a>
<br>
<h3><a id="bp"></a>How Does Backprop Work?</h3>
{:refdef: style="text-align: center;"} 
![](/assets/img/post_img/BP1.JPG){:height="50%" width="50%"}
<br>
*Figure 1. A simple fully connected network. (Image Source: Prof. [Xifeng Yan](https://sites.cs.ucsb.edu/~xyan/) CS165b Slides)* 
{: refdef}

Let us start with some basic math notations. The diagram above is a fully connected neural network with $$ L-1 $$ hidden layers and the output layer $$a^{(L)}$$. The input vector $$ X $$ is an $$N$$-by-$$1$$ vector. $$ W^{(i)} $$ represents the matrix multiplied by the feedforward input at each layer $$i$$, while $$ b^{(i)} $$ is a bias vector added at each hidden layer. $$ a^{(i)} $$ is an $$M$$-by-$$1$$ vector storing each hidden neuron at layer $$i$$. At last, $$ Y^{(k)} $$ represents the desired value of a single output unit $$k$$ inside the $$K$$-by-$$1$$ label vector(from the training samples). Note that in many cases, $$dim(a^{(i)})$$ may not necessarily equals $$dim(X)$$ and $$dim(a^{(i)})$$ can vary throughout the layers.

At each layer $$i$$, the feedforward input $$x^{(i)}$$ or $$a^{(i)}$$ was multiplied by its corresponding matrix $$W^{(i)}$$ and added by the bias vector $$b^{(i)}$$, and we denote this result as $$z^{(i)}$$. This $$z^{(i)}$$ will be then passed into a non-linear activation function $$f$$ (often [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) or [softmax](https://en.wikipedia.org/wiki/Softmax_function)) and gives the values of the hidden vector $$a^i$$, which is fed forward as input for the next layer. Let's dig in deeper into the math behind it. 

At the $$1^{st}$$ layer, we have:
<div style="text-align:center;">
$$
\begin{align*}
  & z^{(1)} = W^{(1)}·X + b^{(1)}\\
  & a^{(1)} = f(z^{(1)}) \\
\end{align*}
$$
</div>

At the $$2^{nd}$$ layer, we have:
<div style="text-align:center;">
$$
\begin{align*}
  & z^{(2)} = W^{(2)}·a^{(1)} + b^{(1)}\\
  & a^{(2)} = f(z^{(2)}) \\
  & ...
\end{align*}
$$
</div>

The equation writes similarly for the rest of the hidden layers. Then at the output layers, we can have:

<div style="text-align:center;">
$$
\begin{align*}
  & z^{(L)} = W^{(L)}·a^{(L-1)} + b^{(L)}\\
  & a^{(L)} = f(z^{(L)}) \\
\end{align*}
$$
</div>

Then, we would like to calculate the Total Error using **Mean Squared Error (MSE)** function $$E$$ and sum up the errors over all the output nodes (About why using the MSE, see [here](https://en.wikipedia.org/wiki/Mean_squared_error#In_regression)), 

<div style="text-align:center;">
$$
\begin{align*}
  & E = \frac{1}{2K} \sum_{k=1}^K (Y^{(k)} - a^{(L)}_{k})^2 + \frac{\lambda}{2} \sum_{l=1}^L (W^{(l)})^2\\
\end{align*}
$$
</div>

where the first term is the **MSE** and the second one an optional [regularization](https://towardsdatascience.com/understanding-the-scaling-of-l²-regularization-in-the-context-of-neural-networks-e3d25f8b50db) term.

<br>
****Important!**** Keep in mind that our ultimate goal is to calculate $$\frac{\delta E}{\delta W_{ij}^{l}}$$, the derivative of error function $$E$$ with respect to an arbitrary element at row $$i$$ and column $$j$$ in an arbitrary matrix $$W^{(l)}$$ at layer $$l$$. 

Denote the error of a single output unit $$k$$ as: 
<div style="text-align:center;">
$$H_k = \frac{1}{2}(Y^{(k)} - a_k^{(L)})^2$$
</div>

As a result, the Total Error can be re-written as:
<div style="text-align:center;">
$$
\begin{align*}
  & E = \frac{1}{K} \sum_{k=1}^K H_k + \frac{\lambda}{2} \sum_{l=1}^L (W^{(l)})^2\\
\end{align*}
$$
</div>

and its derivative as:
<div style="text-align:center;">
$$
\begin{align*}
  & \frac{\delta E}{\delta W_{ij}^{l}} = \frac{1}{K} \sum_{k=1}^K \frac{\delta H_k}{\delta W_{ij}^{l}} + {\lambda}W_{ij}^{l}\\
  \tag{1}
\end{align*}
$$
</div>

The rest of the job is then to calculate $$\frac{\delta H_k}{\delta W_{ij}^{l}}$$. And we can decompose it as: 
<div style="text-align:center;">
$$
\begin{align*}
  &\frac{\delta H_k}{\delta W_{ij}^{l}} = \frac{\delta H_k}{\delta z_{i}^{l}} · \frac{\delta z_{i}^{l}}{\delta W_{ij}^{l}}
  \tag{2}
\end{align*}
$$
</div>

and denote:
<div style="text-align:center;">
$$
\begin{align*}
  & \delta_i^{(l)} = \frac{\delta H_k}{\delta z_{i}^{l}}
  \tag{3}
\end{align*}
$$
</div>
<br>

Okay! Now let's first start with an example to compute the derivative of $$H$$ w.r.t the output unit $$i$$ at the **output layer** $$L$$ using the chain rule:
<div style="text-align:center;">
$$
\begin{align*}
  \delta_i^{(L)} &= \frac{\delta H_i}{\delta z_{i}^{L}} = \frac{\delta}{\delta z_{i}^{L}} \frac{1}{2}(Y^{(i)} - a_{i}^{(L)})^2 \\
  & = -(Y^{(i)} - a_{i}^{(L)}) · \frac{\delta}{\delta z_{i}^{L}} a_{i}^{(L)} \\
  & = -(Y^{(i)} - a_{i}^{(L)}) · f'(z_i^{(L)})
\end{align*}
$$
</div>

And we can use $$\delta^{(L)}$$ to denote the error vector containing all these single output error. **Bear in mind** that this $$\delta^{(L)}$$ is the term that will propagate backward into the network and help us calculate the derivatives of $$H$$ w.r.t to the hidden units! 

Now we can extend the the calculation of $$\delta_i^{(l)}$$ to the hidden layers. Because the error flow is propagating backward(or leftward), we can write the equation for $$\delta_i^{(l)}$$ in terms of the error from the next layer $$\delta^{(l+1)}$$. (This, in fact, is [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming))
<div style="text-align:center;">
$$
\begin{align*}
  \delta_i^{(l)} &= \frac{\delta H_i}{\delta z^{l+1}} \frac{\delta z^{l+1}}{\delta z_{i}^{l}} \\
  & = (W^{(l+1)})^T_i · \delta^{(l+1)} · f'(z_i^{(l)})\\
  \tag{4}
\end{align*}
$$
</div>

This equation might seem frightening at first sight. However, as presented by the figure below, because $$\delta_i^{(l)}$$ is influenced by errors propagating backward from all the units in the next layer, indicated by the red arrows, we will have to take all of these units into account, denoted $$\delta^{(l+1)}$$. So the $$i^{th}$$ row of the transpose of $$W^{(l+1)}$$ will be the weights that the errors are moving through. 

{:refdef: style="text-align: center;"} 
![](/assets/img/post_img/BP2.JPG){:height="50%" width="50%"}
{: refdef}

Nice! Then very simply:
<div style="text-align:center;">
$$
\begin{align*}
  \frac{\delta z_{i}^{l}}{\delta W_{ij}^{l}} = a_j^{(l-1)}
  \tag{5}
\end{align*}
$$
</div>

Plugging eq.$$(4)$$ and $$(5)$$ back to the $$(2)$$ will give us the following:
<div style="text-align:center;">
$$
\begin{align*}
  &\frac{\delta H_k}{\delta W_{ij}^{l}} = (W^{(l+1)})^T_i · \delta^{(l+1)} · f'(z_i^{(l)}) · a_j^{(l-1)}
  \tag{6}
\end{align*}
$$
</div>

Finally, if we plug in eq.$$(6)$$ back to (1), we will obtain the final equation for $$\frac{\delta E}{\delta W_{ij}^{l}}$$, which will be used in **gradient descent** to update the weight at each training epoch:
<div style="text-align:center;">
$$
\begin{align*}
  & W_{ij}^{l} = W_{ij}^{l} + \eta · \frac{\delta E}{\delta W_{ij}^{l}}
  \tag{7}
\end{align*}
$$
</div>
<h3><a id="train_bp"></a>How to train in Backpropagation?</h3>
We will use this wonderful GIF below to explain how to train a network using Back-propagation. 
{:refdef: style="text-align: center;"} 
![](/assets/img/post_img/BP3.gif){:height="80%" width="80%"}
<br>
*Figure 3. Forward and Backward Pass of Back-propagation([Image Source](https://machinelearningknowledge.ai/animated-explanation-of-feed-forward-neural-network-architecture/)))* 
{: refdef}

For each training sample from the dataset, say $$T_0$$, we will feed the input vector into the network to perform a forward pass. After we obtain the predicted output, we will calculate the error(loss) function and perform a backward pass to update each weight in the matrix in each layer. After all weights are updated, we will then enter the next training epoch and feed in the next sample $$T_1$$. The process repeats itself till we exhaust the training set.

