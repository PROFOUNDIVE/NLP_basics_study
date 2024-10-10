## 3.1 N-Grams
history($h$) 다음에 단어 $w$가 이어서 등장할 확률은 다음과 같이 구할 수 있다.

$$
\begin{align}
P(\text{blue} | \text{The water of Walden Pond is so beautifully})=\\ 
\frac{C(\text{The water of Walden Pond is so beautifully bue})}{C(\text{The water of Walden Pond is so beautifully})}
\end{align}
$$

찾고자 하는 문장과 정확히 일치하는 문장은 훈련 데이터셋(Corpus)에 포함되지 않았을 가능성이 높다. 즉, 위와 같은 방식으로 확률을 계산하는 것은 **현실적으로 어렵다**.

하나의 문장은 여러 단어들의 Sequence로 이루어져 있으며, 등장 확률은 각 단어들의 **Joint Probability**를 통해 구할 수 있다. 이는 **Chain Rule**을 통해 다음과 같이 계산된다.

$$\begin{align}
P(w_{1:n}) &= P(w_1)P(w_2|w_1)P(w_3|w_{1:2})...P(w_n|w_{1:n-1}) \\
&= \prod_{k=1}^{n}P(w_k|w_{1:k-1})
\end{align}$$

위 등식에서 $P(w_n|w_{1:n-1})$은 처음에 구하려던 확률과 같다. 즉, 여전히 계산이 어렵다.

### 3.1.1 The Markov assumption
전체 history($h$)를 계산하는 대신,
**마지막 N개의 단어**만을 사용하여 확률을 계산하는 방법이다.

$$
P(w_n|w_{1:n-1}) \approx P(w_n|w_{n-N+1:n-1})
$$

### 3.1.2 How to estimate probabilities: MLE
특정 n-gram의 등장 빈도를 해당 맥락($h$)의 전체 빈도로 나눠서 확률을 구할 수 있다.
- digram: $P(w_n|w_{n-1}) = \frac {C(w_{n-1}w_n)} {C(w_{n-1})}$
- n-gram: $P(w_n|w_{n-N+1:n-1})=\frac {C(w_{n-N+1}w_n)} {C(w_{n-N+1:n-1})}$
### 3.1.3 Dealing with scale in large n-gram models

#### Log Probabilites
확률(0~1)을 계속해서 곱하면 **numerical underflow** 발생 가능
➡️ **log space**에서 확률 계산하여 값이 너무 작아지지 않게 함.

#### Longer context
large n-gram datasets, Infini-gram, quantization, pruning, ...

## 3.2 Evaluating Language Models: Training and Test Sets
언어 모델의 성능 평가에는 주로 Training, Dev, Test 데이터셋이 필요하다.
- Training Set: 모델 학습에 사용되는 데이터셋
- Test Set: 새로운 데이터로 모델 성능 평가. 오버피팅 방지.
## 3.3 Evaluating Language Models: Perplexity
$$
\text{perplexity}(W)= \sqrt[N]{\prod_{i=1}^{N}{\frac{1}{P(w_i|w_1...w_{i-1})}}}
$$
## 3.6 Smoothing, Interpolation, and Backoff
Peplexity는 확률의 역수이므로, 찾고자 하는 단어 또는 문장이 데이터셋에 없을 경우(확률이 0) 계산이 불가능하다.
### 3.6.1 Laplace Smoothing
count에 1씩 더해서 확률이 0이 되지 않도록 한다. (add-one smoothing)

$$
P_{\text{Laplace}}(w_i) = \frac{c_i + 1}{N + V}
$$
### 3.6.3 Language Model Interpolation
### 3.6.4 Stupid Backoff
$$
S(w_i|w_{i-N+1:i-1}) = 
\begin{cases} 
\frac{\text{count}(w_{i-N+1:i})}{\text{count}(w_{i-N+1:i-1})} & \text{if } \text{count}(w_{i-N+1:i}) > 0 \\
\lambda S(w_i|w_{i-N+2:i-1}) & \text{otherwise}
\end{cases}
$$
