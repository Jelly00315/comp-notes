# COMP 4471 / ELEC 4240 — Lecture 2 Summary

## Image Classification with KNN and Linear Classifiers

## 1. Image classification

Image classification assigns an input image to one label from a fixed set of classes.

The main difficulty is the **semantic gap**: humans see meaningful objects, while a computer initially sees only an array of pixel values. Recognition must remain reliable despite viewpoint changes, illumination, background clutter, occlusion, deformation, context, and variation within the same class.

Because explicit rules for recognizing every object are impractical, computer vision uses a **data-driven approach**:

1. Collect images and labels.
2. Train a classifier using the labelled data.
3. Evaluate it on unseen images.

---

## 2. Nearest-neighbor classifiers

### 2.1 Nearest neighbor

The nearest-neighbor classifier memorizes the training data. For a new image (x), it finds the training image (x_i) with minimum distance and copies its label:

$$
i^*=\arg\min_i d(x,x_i), \qquad \hat y=y_{i^*}.
$$

Common distance metrics are:

$$
d_{L1}(x,z)=\sum_p |x_p-z_p|
$$

and

$$
d_{L2}(x,z)=\sqrt{\sum_p(x_p-z_p)^2}.
$$

Training involves almost no computation, but prediction requires comparison with all (N) training examples. For images with (D) pixel values, prediction costs approximately (O(ND)).

### 2.2 K-nearest neighbors

KNN selects the (K) closest training examples and predicts their majority label:

$$
\hat y=\operatorname{mode}\{y_{i_1},\ldots,y_{i_K}\}.
$$

- Small (K): sensitive to noise.
- Large (K): smoother, but distant and irrelevant examples may influence the prediction.
- (K) and the distance metric are **hyperparameters**.

### 2.3 Choosing hyperparameters

- **Training set:** learns parameters or supplies stored KNN examples.
- **Validation set:** selects hyperparameters such as (K) and the distance metric.
- **Test set:** evaluates the completed model once at the end.

Using the test set to choose hyperparameters makes the final performance estimate biased.

For small datasets, (K)-fold cross-validation rotates the validation fold and averages the results. It is less common for large neural networks because repeated training is expensive.

### 2.4 Why raw-pixel KNN is unsuitable for images

1. Pixel distance does not represent semantic similarity. A small translation may change many pixels without changing the object.
2. Images are extremely high-dimensional. Data becomes sparse as dimension increases, so even the nearest example may be far away. This is the **curse of dimensionality**.

---

## 3. Parametric approach and linear classifiers

A parametric classifier compresses what it learns into a fixed set of parameters instead of storing the full training set.

The linear classifier is:

$$
f(x;W,b)=Wx+b.
$$

For CIFAR-10:

$$
x\in\mathbb R^{3072},\qquad
W\in\mathbb R^{10\times3072},\qquad
b\in\mathbb R^{10}.
$$

The model produces ten class scores:

$$
s=Wx+b.
$$

Prediction chooses the largest score:

$$
\hat y=\arg\max_j s_j.
$$

Each row ($w_j^\top$) of (W) can be viewed as a learned template for class (j):

$$
s_j=w_j^\top x+b_j.
$$

Geometrically, equality between two class scores defines a hyperplane. Therefore, a linear classifier can only form linear decision boundaries. It cannot directly represent XOR patterns, rings, or multiple disconnected regions.

---

## 4. Training objective

For labelled examples ((x_i,y_i)), a loss function measures how unsuitable the current scores are. The dataset loss is usually the average:

$$
L(W,b)=\frac{1}{N}\sum_{i=1}^{N}L_i.
$$

Training seeks parameters that minimize this loss:

$$
(W^*,b^*)=\arg\min_{W,b}L(W,b).
$$

During inference, the model only computes scores and takes the argmax. Loss is needed during training because the true labels are available there.

---

## 5. Multiclass SVM loss

For example (i), let (s_{y_i}) be the correct-class score and (s_j) an incorrect-class score. With margin (1):

$$
L_i=\sum_{j\ne y_i}\max(0,s_j-s_{y_i}+1).
$$

~={red}The correct class should beat every incorrect class by at least (1):=~

$$
s_{y_i}\ge s_j+1.
$$

If this condition holds for all incorrect classes, ($L_i=0$). Once the margin is satisfied, increasing the correct score further does not reduce SVM loss.

Important properties:

- Minimum loss: (0).
- No finite maximum.
- If all (C) scores are approximately equal at initialization, ($L_i\approx C-1$).
- Including the correct class in the sum adds the constant (1) to every example.
- Taking a mean instead of a sum only rescales the loss.
- Squaring each hinge penalty gives greater weight to severe violations.

Example with correct class cat and scores ([3.2,5.1,-1.7]):

$$
L_i=\max(0,5.1-3.2+1)+\max(0,-1.7-3.2+1)=2.9.
$$

---

## 6. Softmax classifier and cross-entropy

Raw class scores are called **logits**. Softmax converts them into probabilities:

$$
Q_j=P(Y=j\mid X=x_i)=\frac{e^{s_j}}{\sum_k e^{s_k}}.
$$

Every (Q_j\ge0), and the probabilities sum to (1).

If the true label is represented by a one-hot distribution (P), cross-entropy is:

$$
H(P,Q)=-\sum_j P_j\log Q_j.
$$

Because (P) is one-hot, it selects the predicted probability of the correct class:

$$
L_i=-\log Q_{y_i}
=-\log P(Y=y_i\mid X=x_i).
$$

The average dataset loss is:

$$
L=-\frac1N\sum_{i=1}^{N}\log P(Y=y_i\mid X=x_i).
$$
A high correct-class probability produces a small loss; a probability near zero produces a very large loss.

Important properties:

- Minimum loss approaches (0).
- No finite maximum.
- Equal initial scores give ($Q_j=1/C$), so ($L_i=\log C$).
- For CIFAR-10, the expected initial loss is ($log 10\approx2.3$).

Minimizing cross-entropy is equivalent to maximum-likelihood estimation: it maximizes the probability assigned to the observed correct labels.

Cross-entropy also satisfies:

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q).
$$

Since the target distribution (P) is fixed, minimizing cross-entropy also minimizes the KL divergence between the target and predicted distributions.

---

## 7. SVM versus softmax

| Aspect                      | Multiclass SVM                       | Softmax cross-entropy                         |
| --------------------------- | ------------------------------------ | --------------------------------------------- |
| Goal                        | Correct score wins by a fixed margin | Correct class receives high probability       |
| After sufficient separation | Loss becomes exactly (0)             | Loss continues decreasing as confidence rises |
| Output interpretation       | Relative scores                      | Probability distribution after softmax        |

For correct class (0), both score vectors ([10,9,9]) and ([10,-100,-100]) have zero SVM loss. Softmax strongly prefers the second because it assigns much greater probability to the correct class.

---

## 8. Complete learning pipeline

### Training

$$
x_i\rightarrow Wx_i+b\rightarrow\text{scores}\rightarrow\text{loss}\rightarrow\text{gradient}\rightarrow\text{update }W,b.
$$

Gradient descent updates the parameters using:

$$
W\leftarrow W-\eta\frac{\partial L}{\partial W},
\qquad
b\leftarrow b-\eta\frac{\partial L}{\partial b},
$$

where (eta) is the learning rate.

### Inference

$$
x\rightarrow Wx+b\rightarrow\arg\max_j s_j.
$$

No loss is required during ordinary prediction because the true label is unknown.

---

## Key takeaway

The lecture introduces the fundamental supervised-classification framework:

$$
\boxed{\text{The score function makes predictions; the loss function evaluates and trains it.}}
$$

KNN predicts using stored examples. A linear classifier instead learns parameters (W,b), produces class scores, and is trained using SVM or softmax cross-entropy loss. The following lecture continues with regularization and optimization.
