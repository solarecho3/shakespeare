# shakespeare
An InstructGPT-based model to talk to the bard

## goal

To produce a GPT, trained on all shakespeare, that is queriable in plain English. We intend for this model to be able to answer an imperative natural language prompt such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt."

## method

The first step will be to design a deep reinforcement learning with human feedback (RLHF) system, the aim of which is to design a reward function. That reward function will then be used to optimize to reward function of more traditional unsupervised reinforcement learning models.

## lexical definitions

### variables
$t$ = time

$o$ = observation

$O$ = set of observations

$a$ = action

$A$ = set of actions

$\succ$ = indicates preference

$\tau$ = trajectories

$\sigma$ = trajectory segments

$\pi$ = policy

$\hat{r}$ = reward function estimate

$\Bbb R$ = set of rewards

References: https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference

### method

1. Agent $\Delta$ polls the environment for observations in time $t$

2. At each time $t$, $\Delta$ receives an observation $o_t\in O$

3. $\Delta$ uses policy $\pi$ to form and send an action $a_t\in A$

4. Each $o_t$ and $a_t$ form a sequence of (observation, action) tuples in a tracjectory segment $\sigma$, where $\sigma=((o_0,a_0),(o_1,a_1),...,(o_{k-1},a_{k-1}))\in(O\times A)^k$

5. Whereas normally a reward function $r_t\in\Bbb R$ is applied, we have human indicate preference, as $\sigma^1\succ\sigma^2$, instead

6. Algorithm behavior evaluation can be either **quantitative**: 

    Preferences $\succ$ are generated by a reward function $r:O\times A\rightarrow\Bbb R$ if:

    $((o_0^1,a_0^1),...,(o_{k-1}^1,a_{k-1}^1))\succ((o_0^2,a_0^2),...,(o_{k-1}^2,a_{k-1}^2))$

    whenever

    $r((o_0^1,a_0^1) + ... + (o_{k-1}^1,a_{k-1}^1)) > r((o_0^2,a_0^2),...,(o_{k-1}^2,a_{k-1}^2))$

7. Or **qualitative**:

    At each time $t$ the policy $\pi : O \rightarrow A$, and a reward function estimate $\hat{r}: O \times A \rightarrow\Bbb R$, each parameterized by deep neural networks


## references
[https://openai.com/blog/instruction-following/]
[https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/]
[https://github.com/openai/following-instructions-human-feedback]
