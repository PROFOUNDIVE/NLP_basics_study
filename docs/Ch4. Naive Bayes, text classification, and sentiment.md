## Naive Bayes, Text classification, and Sentiment
### Contents
4.1. Naive Bayes classifiers  
4.2. Training the Naive Bayes classifier  
4.3. Worked example  
4.6. Naive Bayes as a Language Model  
4.7. Evaluation: Precision, Recall, $F$-measure  
Many language processing tasks can be viewed as tasks of classification.

## Background
**Tasks we can use clasification**
1. text categorization
2. sentiment analysis
3. spam detection

**probablistic classifier**

We can view of a supervised classifier as
$f:x\in X\rightarrow y\in Y=\\{y_1,y_2,\cdots,y_M\\}$  
where $x:\text{input}, y_i:\text{predicted class},$ $X:\text{inputs}, Y:\text{classes}$  
For text classification, instead we say
$f:d\in D\rightarrow c\in C$, where $d:\text{document}, c:\text{class}$  

In the supervised situation, we have a training set $\left\\{\left(d_{1}, c_{1}\right), \ldots,\left(d_{N}, c_{N}\right)\right\\}$ .  
Goal: To learn a classifier that is capable of mapping $f$.

A probabilistic classifier additionally will tell us the probability of the observation being in the class.  

**Two kinds of Classifier**
1. Generative classifier
2. Discriminative classifier

| Type | Objective | Examples |
|---|---|---|
| Generative | $P(x{\mid}y),P(y)$ | Naive Bayes |
| Discriminative | $P(y{\mid}x)$ | Logistic Regression, SVM, NN |

## 4.1. Naive Bayes Classifiers

**naive Bayes classifier**
Naive Bayes is a **probabilistic classifier**
$f:d\in D\rightarrow c\in C\ (d\mapsto \hat{c})$

In this,
$$\hat{c}=\underset{c \in C}{\text{argmax}} P(c \mid d)\quad\cdots\quad(4.1.)$$

$\hat{c}$ : 주어진 document에서 output $C$ 중 가장 나올 확률이 높은 class.  

$\hat{}$ : estimated


Now we will use Bayes' rule (bayesian inference)
$$P(x \mid y)=\frac{P(y \mid x) P(x)}{P(y)} \quad\cdots\quad({4.2})$$

By using this, we can compute $P(x\mid y)$ indirectly, even though we don't know about $P(x\mid y)$ .

We can then substitute Eq. 4.2 into Eq. 4.1 to get:

$$\hat{c}=\underset{c \in C}{\text{argmax}} P(c \mid d)=\underset{c \in C}{\text{argmax}} \frac{P(d \mid c) P(c)}{P(d)}$$

For convention, we drop the denominator $P(d)$ .
This is possible because we aim to find $c$ in which $P(c|d)$ is maximized for each class, where $P(d)$ is common.  
Thus, we can choose the class that maximizes this simpler formula:

$$\hat{c}=\underset{c \in C}{\text{argmax}} P(c \mid d)=\underset{c \in C}{\text{argmax}} P(d \mid c) P(c)$$

By $P(d|c)$ term, NB classifier is Generative.

We compute the prob. that document $d$ comes for given class $c$ .  

$$\hat{c}=\underset{c \in C}{\text{argmax}} \overbrace{P(d \mid c)}^{\text {likelihood }} \overbrace{P(c)}^{\text {prior }}$$


WLOG, we can represent a document $d$ as a set finite of features $f_{1}, f_{2}, \ldots, f_{n}:$
$$\hat{c}=\underset{c \in C}{\text{argmax}} \overbrace{P\left(f_{1}, f_{2}, \ldots, f_{n} \mid c\right)}^{\text {likelihood }} \overbrace{P(c)}^{\text {prior }}$$


The equation is still **too hard to compute directly**:
without some simplifying assumptions, estimating the probability of every possible combination of features would require huge numbers of parameters and impossibly large training sets.  


Naive Bayes classifiers therefore make two simplifying assumptions.
1. bag-of-words: positions don't matter, the counts only matter.
2. naive Bayes assumption: $P\left(f_{i} \mid c\right)$ are independent so can be '**naively**' multiplied as follows;  
$P\left(f_{1}, f_{2}, \ldots ., f_{n} \mid c\right)=\\ P\left(f_{1} \mid c\right) \cdot P\left(f_{2} \mid c\right)  \cdots P\left(f_{n} \mid c\right)$

The equation for the class chosen by a NB classifier is thus:

$$c_{N B}=\underset{c \in C}{\text{argmax}} P(c) \prod_{f \in F} P(f \mid c)$$

Note. $\hat{c}\neq C_{NB}$ , $\hat{c}\approx C_{NB}$


**Application**. Naive Bayes for text classification

$$\text { positions } \leftarrow \text { all word positions in test document }$$
$$c_{N B} =\underset{c \in C}{\text{argmax}}\ P(c) \prod_{i \in \text { positions }} P\left(w_{i} \mid c\right)$$

$w_i$ : $i$ -th word as a feature


To avoid underflow and increase speed, we will do this on log space. Thus the eq. is generally instead expressed as

$$c_{N B}=\underset{c \in C}{\text{argmax}} \log P(c)+\sum_{i \in \text { positions }} \log P\left(w_{i} \mid c\right)$$


By considering features in log space, Eq. 4.10 computes the predicted class as a linear function of input features.

Classifiers that use a linear combination of the inputs to make a classification decision -like naive Bayes and also logistic regressionlinear are called linear classifiers.  

## 4.2. Training the Naive Bayes Classifier
$$\hat{P}(c)=\frac{N_{c}}{N_{d o c}}$$

$$\hat{P}\left(w_{i} \mid c\right)=\frac{\text{count}\left(w_{i}, c\right)}{\sum_{w \in V} \text{count}(w, c)}$$

$V$ : the union of all the word types in all classes, not just the words in one class $c$ .  

e.g. 
$\hat{P}(\text { "fantastic" } \mid \text { positive })=\dfrac{\text{count}(\text { "fantastic", positive })}{\sum_{w \in V} \text{count}(w, \text { positive })}$  


But since naive Bayes naively multiplies all the feature likelihoods together, zero probabilities in the likelihood term for any class will make $\hat{P}=0$ .

The simplest solution is the add-one (Laplace) smoothing, **simple and powerful** smoothing in Naive Bayes.

$\hat{P}\left(w_{i} \mid c\right)=\dfrac{\text{count}\left(w_{i}, c\right)+1}{\sum_{w \in V}(\text{count}(w, c)+1)}=\dfrac{\text{count}\left(w_{i}, c\right)+1}{\left(\sum_{w \in V} \text{count}(w, c)\right)+|V|}$  


**unknown word**

What do we do when we first see in test data, but we didn't see at all because they did not occur in any training document in any class?
How can we compute $\text{count}(w_i,c)$?

**Solution**. For such unknown words is to ignore them, remove them from the test document, don't include that in our computation


**stop words**

We can ignore very frequent words like $the, a$ .

But it just slightly changes the performance, we don't adopt it.


## 4.3. Worked example
An example for computing naive Bayes classifier
![image.png](../assets/image_1728567730868_0.png)
![image.png](../assets/image_1728567760779_0.png)


Pseudo code of add-1(Laplace) smoothing NB
```latex
**function** Train Naive Bayes(D, C) **returns** $V, \log P(c), \log P(w \mid c)$
**for each** class $c \in C \quad$ # Calculate $P(c)$ terms  
$\mathrm{N}_{d o c}=$ number of documents in D  
$\mathrm{N}_{c}=$ number of documents from D in class c  
logprior $[\mathrm{c}] \leftarrow \log \frac{N_{c}}{N_{d o c}}$  
$V \leftarrow$ vocabulary of D  
bigdoc $[c] \leftarrow$ **append**(d) **for** $\mathrm{d} \in \mathrm{D}$ **with** class $c$  
**for each** word $w$ in V # Calculate $P(w \mid c)$ terms  
$\text{count}(w, c) \leftarrow \#$ of occurrences of $w$ in bigdoc $[c]$  
loglikelihood $[\mathrm{w}, \mathrm{c}] \leftarrow \log \frac{\text{count}(w, c)+1}{\sum_{w^{\prime} \text { in } V}\left(\text{count}\left(w^{\prime}, c\right)+1\right)}$  
$\mathbf{r n}$ logprior, loglikelihood, $V$  
**function** Test NAIVE BAYES(testdoc, logprior, loglikelihood, $\mathrm{C}, \mathrm{V}$ ) **returns** best $c$  
**for each** class $c \in C$  
sum $[c] \leftarrow$ logprior $[c]$  
**for each** position $i$ in testdoc  
word $\leftarrow$ testdoc $[i]$  
**if** word $\in V$  
  $\text{sum}[c] \leftarrow \text{sum}[c]+$ loglikelihood $[$ word, $c]$  
return $\text{argmax}_{c} \text{sum}[c]$
```

## 4.6. Naive Bayes as a Language model
Since the likelihood features can assign a probability to each word $P(\text{word}\mid c)$ , the model also assigns a probability to each sentence:

$$P(s \mid c)=\prod_{i \in \text { positions }} P\left(w_{i} \mid c\right)$$

example
![image.png](../assets/image_1728570945734_0.png){:height 343, :width 689}

## 4.7. Evaluation: Precision, Recall, $F$ -measure

To evaluate NB classifier just using accuracy is ~~bullshit~~. why?

**Motivation**.
Imagine a simple tweet classifier that stupidly classified every tweet as "not related".  
This classifier would have 999,900 true negatives and only 100 false negatives for an accuracy of $99.99 \%$ ! What an amazing accuracy level!  
Surely we should not use accuary to evaluate.  


system output labels: 결과값. expected labels

gold standard labels: 정답. human labels  

![](https://cdn.mathpix.com/cropped/2024_10_06_fbf3796c5fa5e5bfe9a7g-12.jpg?height=429&width=1167&top_left_y=368&top_left_x=487)
confusion matrix  

$$\text { Precision }=\frac{\text { true positives }}{\text { true positives }+ \text { false positives }}$$
$$\text { Recall }=\frac{\text { true positives }}{\text { true positives }+ \text { false negatives }}$$

There are many ways to define a single metric that incorporates aspects of both precision and recall.

The simplest of these combinations is the $F$ -measure (van Rijsbergen, 1975):  
$$F_{\beta}=\dfrac{\left(\beta^{2}+1\right) P R}{\beta^{2} P+R}$$

$\beta>1$ : favor recall  
$\beta<1$ : favor precision  
$\beta=1$ : equally favor  


#### 4.7.1. Evaluating with more than two classes
Up to now we have been describing text classification tasks with **only two classes**.

But lots of classification tasks in language processing have more than two classes, even higher than a hundread.  
Luckily the naive Bayes algorithm is already a multi-class classification algorithm.
We can just add some features to classify.  


But how can we evalute then?  
We can't adapt precision, recall directly.  


![image.png](../assets/image_1728572650857_0.png)
**Simply we think of precisions and recalls for each classes.**  

And there are two ways of computing precisions/recalls
![image.png](../assets/image_1728572834258_0.png)
