"""
What is a neural network ?

a neural network is a mathematical function that learns
input -> output



hours studied ──┐
                ├──> neural network ──> exam score
sleep hours ────┘




the neural network discovers the mathematical relationship between the inputs
and outputs

What is a neuron ?
a neuro takes numbers multiplies them by weights and adds a bias
and producesa  number

z = x1w1 + x2w2 + b

w : weight
b : bias


why weights ?
weights determine importance

imagine predicting whether someone will pass
hours studied → important
hours gaming   → maybe negatively important
The model might learn:
study weight = +2.5
gaming weight = -1.3
So:
score=2.5(study)−1.3(gaming)+b
The neural network's job is largely to learn good weights and biases.
This is the central idea.
Neural networks learn parameters (weights and biases) that transform inputs into useful predictions.


where does the "neural" part come from ?
one neuron isn't very interesting
connect many neurons

Inputs

x₁ ─────┐
x₂ ─────┼──> neuron
x₃ ─────┘

now multiple neurons:
             ┌── neuron 1
x₁ ──────────┼── neuron 2
x₂ ──────────┼── neuron 3
x₃ ──────────┘
that's a layer

Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output



BUT THERE IS  A PROBLEM

Suppose we only do:
    z = Wx + b

and stack many layers :
 x
 ↓
Wx+b
 ↓
Wx+b
 ↓
Wx+b
 ↓
output

Mathematically, a sequence of linear transformations is still basically another linear transformation.
So the network would be surprisingly limited.
We need non-linearity.



1. What is a linear transformation?
A linear transformation is basically a rule that takes numbers in and changes them in a predictable, straight-line way.
For example:
y=2x
If:
x=1 → y=2
x=2 → y=4
x=3 → y=6
It's just scaling.
A neural-network neuron before activation does something like:
z=wx+b
Again: multiply by a weight, then add a bias.


2. Here's the important part
Imagine we build a network with several layers:
x→Layer 1→Layer 2→Layer 3
You might think:
"Wow, three layers! This must let the network do really complicated things."
But if every layer only performs linear operations, mathematically they can be collapsed into one operation.
For example:
x→2x→3(2x)→5(3(2x))
is simply:
30x
So despite having three layers, the whole thing is equivalent to:
y=30x
That's not very powerful.


why its bad in real world relationships are not straight lines
imagine trying to classify points on an axis




4. Non-linearity is the trick
So after doing:
z=wx+b
we apply an activation function:
a=f(z)
For example, ReLU:
ReLU(x)=max(0,x)
It basically says:
negative → turn it into 0
positive → leave it alone
So:
−5→0
2→2
10→10
That little operation breaks the purely linear nature of the network.
Now we can have:
x→(wx+b)→ReLU→(wx+b)→ReLU→...
And stacking these nonlinear transformations allows the network to build increasingly complicated functions.


The simplest mental model
Without activation:
Layer + Layer + Layer = effectively one big linear operation.
With activation:
Layer → nonlinear bend → Layer → nonlinear bend → Layer
Now the network can learn curves, boundaries, patterns, and complicated relationships.
That's why when learning neural networks, you should remember:

Weights learn transformations. Activation functions give the network the ability to represent complex, non-linear relationships.

actual flow :


random weights
      ↓
    forward
      ↓
  prediction
      ↓
     loss
      ↓
backpropagation
      ↓
 gradients
      ↓
 update weights
      ↓
    repeat







FORWARD PROPOGATION :
Forward propagation = take the input and push it through the network until you get a prediction.


Why do we need a loss?
Suppose our network is trying to predict house prices.
Actual price:
₹50 lakh
Network predicts:
₹42 lakh
Clearly, it fucked up.
We need a number that tells us how badly it fucked up.
That's the loss.


For simple regression, we can use MSE:
MSE=(prediction−actual)^2
2

So:
MSE=(42−50)
2

=64
Loss = 64.
The bigger the mistake, the bigger the loss.
Our goal:
make loss as small as possible


GRADIENT DESCENT:
Now we actually change the weight.
We moved the weight up.
So remember:
positive gradient → decrease weight
negative gradient → increase weight
That's gradient descent.



BACK PROPAGATION:
We have potentially millions/billions of weights.
We need to know:
"What is the gradient of the loss with respect to EVERY weight?"
Backpropagation calculates that.
It uses the chain rule from calculus.
You don't need to panic about calculus yet.
Conceptually:
prediction
    ↓
   loss
    ↓
Which parameters affected the loss?
    ↓
Calculate gradients
    ↓
Update parameters
So:
Forward propagation
"What prediction do I get?"
Loss
"How wrong am I?"
Backpropagation
"Which weights are responsible, and by how much?"
Gradient descent
"Okay, change those weights."




Forward propagation
"What prediction do I get?"
Loss
"How wrong am I?"
Backpropagation
"Which weights are responsible, and by how much?"
Gradient descent
"Okay, change those weights."



          FORWARD
             ↓
        input → prediction
             ↓
            LOSS
             ↓
       "How wrong?"
             ↓
       BACKPROPAGATION
             ↓
          gradients
             ↓
       GRADIENT DESCENT
             ↓
        update weights
             ↓
          repeat
"""



x = 2

target = 10

w = 3.0  # weight
learning_rate = 0.1

for step in range(10):
    # forward propagation
    prediction = x * w

    # loss function
    loss = (prediction - target) **2

    # backpropagation
    dloss_dprediction = 2 * (prediction - target )
    dprediction_dw = x

    gradient= dloss_dprediction * dprediction_dw


    #updation

    w = w - learning_rate * gradient
    print(
        f"step={step}, "
        f"prediction={prediction:.4f}, "
        f"loss={loss:.4f}, "
        f"gradient={gradient:.4f}, "
        f"w={w:.4f}"
        )

