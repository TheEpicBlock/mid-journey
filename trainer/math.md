This is a summary of the neural network used here as well as the math used in the process. This is one of the simplest forms of neural networks you can make. This document should help explain some of the code, but it primarily serves as a reminder to me of what everything is supposed to do.

The neural network consists of a number of layers. Each layer has a number of nodes where each node is connected with each node in the previous layer. I recommend [3b1b's series on the topic](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) if this sounds foreign to you. I specifically rewatched the two parts on backpropagation to implement my network, they're a good watch.

Each node in the network will have a certain activation value. This will be indicated with $`a^{(L)}_i`$ (matching 3b1b), which is the i-th node of layer L. A node's activation is determined by the activation of the previous layer's nodes. Each of these connections has a weight. The weight will be indicated as $`w^{(L)}_{ij}`$, indicating the weight from the i-th node in the previous layer to the j-th node in layer L. Note that this is inverted from the notation in 3b1b's video. Additionally, the node also has its own bias which is added on top. We will combine the weights and biases in an intermediate output, making $z$ is equal to the sum of all weights in the previous layer multiplied by their activation.
```math
z^{(L)}_i = \sum_{k=1}^{n(L-1)}(w^{(L)}_{ki}a^{(L-1)}_k)+b^{(L)}_i
```
Then, the activation is defined by plugging $z$ into an activation function.
```math
a^{(L)}_i = \sigma(z^{(L)}_i)
```
Anything from [this page](https://en.wikipedia.org/wiki/Activation_function) will do (except the identity). I chose the logistic function because 3b1b mentioned that one first and it seemed pretty neat to me. Some of those activation functions are better suited for certain tasks or something.

<details>
<summary>Quick aside on what our input/outputs mean</summary>
For this network, the last layer should contain three nodes containing the output colour (in OkLab, coz we're fancy). For the input, we have a fixed number of nodes. Each character from the input is converted to a number and the rest of the nodes are set to zero. This would be equivalent to padding out the input string to have a fixed number of characters.
</details>

The network's output can be computed by calculating the activations layer by layer until you've reached the last one. During training we want to continually refine our network. We define a cost function: this is equivalent to the sum of the difference between the expected and actual output.
```math
C_0 = \sum_{j=0}^{n(L)}(a^{(L)}_j-e_j)^2
```
With $L$ being defined here as the last layer, and $e$ being our expected values. The goal of training is to minimize the cost function. We won't actually calculate this cost function very often, but it's useful for the math-y stuff.

# Backpropagation

The math-y stuff! Yay! I won't give an *exact* explanation of backpropagation. I'd recommend watching the 3b1b video (from which I got these equations), but in short: you need to view your network as one large equation and compute the derivative with respect to all the variables inside the equation individually. In this case those variables will be the weights and biases. We can use the chain rule for this. Here are some of the equations, they're more legible when put into something other than GitHub:

<details>
<summary>Some of my equations</summary>

Please put these into some latex renderer like [this one](https://latexeditor.lagrida.com/). GitHub doesn't like it.

```
C_0 = \sum_{j=0}^{\abs{L}}(a^{(L)}_j-e_j)^2 \\
z^{(L)}_j = \sum_{k=0}^{|L-1|}(w^{(L)}_{kj}a^{(L-1)}_k)+b^{(L)}_j \\
a^{(L)}_j = \sigma(z^{(L)}_j) \\
\text{We want: the derivative of $C_0$ with respect to $w^{(L)}_{kj}$, $b^{(L)}_j$ and $a^{(L-1)}_k$}\\
\pdv{C_0}{a^{(L)}_h} = 2(a^{(L)}_h-e_h) \\
\pdv{a^{(L)}_h}{z^{(L)}_h} = \sigma'(z^{(L)}_h)\\

\hspace{13cm} \text{Derivative to $w^{(L)}_{kj}$}\\
\pdv{z^{(L)}_j}{w^{(L)}_{kj}}=a^{(L-1)}_k\\
\pdv{C_0}{w^{(L)}_{kh}} = \pdv{z^{(L)}_h}{w^{(L)}_{kh}}\pdv{a^{(L)}_h}{z^{(L)}_h}\pdv{C_0}{a^{(L)}_h} = a^{(L-1)}_k\sigma'(z^{(L)}_h)2(a^{(L)}_h-e_h)\\

\hspace{13cm} \text{Derivative to $b^{(L)}_j$}\\
\pdv{z^{(L)}_j}{b^{(L)}_{j}}=1\\
\pdv{C_0}{b^{(L)}_{h}} = \pdv{z^{(L)}_h}{b^{(L)}_h}\pdv{a^{(L)}_h}{z^{(L)}_h}\pdv{C_0}{a^{(L)}_h} = \sigma'(z^{(L)}_h)2(a^{(L)}_h-e_h)\\

\hspace{13cm} \text{Derivative to $a^{(L-1)}_k$}\\
\pdv{C_0}{a^{(L-1)}_k} = \sum_{h=0}^{n_L-1}\pdv{z^{(L)}_h}{a^{(L-1)}_k}\pdv{a^{(L)}_h}{z^{(L)}_h}\pdv{C_0}{a^{(L)}_h} \\
\pdv{z^{(L)}_j}{a^{(L-1)}_k}=w^{(L)}_{kj} \\
\pdv{C_0}{a^{(L-1)}_k} = \sum_{h=0}^{n_L-1}w^{(L)}_{kh}\sigma'(z^{(L)}_h)2(a^{(L)}_h-e_h)
```

</details>

Follows is the process that the trainer does to apply backpropagation.

Firstly, all the training data is loaded. We compute for each layer the values of $z$ as well as the values of $a$. These are stored in separate buffers. Note that we do the computation of every training value at once. This means that our buffers will actually contain multiple evaluations of our network for multiple inputs. After we've done this step, we should have for each layer a buffer containing $a$ and $z$.

Now we're ready to do backpropagation. Firstly, we compute $`\frac{\partial C_0}{\partial a^{(L)}_i}`$ for the last layer.
This is pretty trivial, it's equal to $`2(A^{(L)}_i-e_i)`$. We get to that function by differentiating our cost function with respect to $`a^{(L)}_i`$.

We can now use the chain rule to compute the derivatives of other values. First the derivative of z is computed ($`\frac{\partial C_0}{\partial z^{(L)}_i}`$). We use the chain rule here to compute it by multiplying $`\frac{\partial C_0}{\partial a^{(L)}_i}`$ (which we calculated in the previous step) by $`\frac{\partial a^{(L)}_h}{\partial z^{(L)}_h} = \sigma'(z^{(L)}_h)`$. Note that this works for any layer, as long as we have the value of $`\frac{\partial C_0}{\partial a^{(L)}_i}`$ for that layer. Later we'll compute $`\frac{\partial C_0}{\partial a^{(L-1)}_i}`$ which means we can invoke this step again.

Okay, so now we know $`\frac{\partial C_0}{\partial z^{(L)}_i}`$. This happens to be equal to $`\frac{\partial C_0}{\partial b^{(L)}_i}`$ so we now know how to adjust the bias. The shader will store this $z$/$b$ value in a buffer for future reference.
The value of $`\frac{\partial C_0}{\partial w^{(L)}_{ij}}`$ isn't that hard to compute, it's $`a^{(L-1)}_i \frac{\partial C_0}{\partial z^{(L)}_i}`$

Now when we want to do the next layer we need $`\frac{\partial C_0}{\partial a^{(L-1)}_i}`$. This one is a little more complicated:

```math
\frac{\partial C_0}{\partial a^{(L-1)}_i} = \sum_{j=1}^{n(L-1)} (w^{(L)}_{ij} \frac{\partial C_0}{\partial z^{(L)}_i})
```

This entails that we need to sum the weights and $`\frac{\partial C_0}{\partial z}`$ values of the $L$ layer to compute the $`\frac{\partial C_0}{\partial a}`$ values for $L-1$. In the code this is implemented by having two different shader entrypoints. One which computes $`\frac{\partial C_0}{\partial z^{(L)}_i}`$ using the cost based formula and one which calculates it using the formula based on the previous layer. Once the derivative with respect to $z$ is found, the process is the same.

# Applying backpropagation

Now there's one final step left to go. We have the derivatives of each of all our parameters. We just need to apply them to improve our network. One tiny thing to take into account: the derivatives have been computed for each *invocation* of our network. We want to average all the derivatives. Then we can simply sum the derivatives with our current weights and biases to improve our network. Repeat this several times to train your network!