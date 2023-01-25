An InstructGPT-based model to talk to the bard

How could a GPT provide accurate, intelligence quality answers to specific prompts by users based on a corpus of provided data?

## goal

To produce a GPT, trained on all shakespeare, that is queriable in plain English. We intend for this model to be able to answer an imperative natural language prompt such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt."

The goal of this research is  to provide a practical implementation of GPT3 that meets the desired end state of the customer.  The result of the research will be a GPT, trained on all Shakespeare's writing that will allow the user to ask plain text questions such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt." The query returned will be a timely, accurate, and truthful response that answers the intent of the users query.  Accurate answers are the paramount priority of the tool.  Answers to queries will  be returned with a relative probability  (between zero (0) and one (1) with one  being 100% accurate) that the returned answer is correct.  

## method

The first step will be to design a deep reinforcement learning with human feedback (RLHF) system, the aim of which is to design a reward function. That reward function will then be used to optimize to reward function of more traditional unsupervised reinforcement learning models.

The tool will utilize mixed methods to maximize the existing qualitative and quantitative algorithms to produce the best product.  The qualitative method will be used to provide human input to train the reward function which will assist in determining how the tool will respond to human inputs.  The resulting reward function will then be applied in an unsupervised quantitative way to train on the corpus of data, in this case the entirety of Shakespeare's written works.  

describe qualitative methods
describe quantitative methods
why did we use them together?

## roadmap
- [ ] Select a GPT for our use case - [InstructGPT](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf)
- [ ] Research Reinforcement Learning from Human Feedback (RLHF)
- [ ] Theorize a practical application of the RLHF method for our use case
- [ ] Theorize python implementation of RLHF and practical use case
- [ ] Search for python implementations of InstructGPT and a similar RLHF method
- [ ] Reading list
- [ ] Let's code - NanoGPT (Andrej Karpathy)
- [ ] Select Reinforcement Learning algorithm [OpenAi baselines](https://github.com/openai/baselines), used to optimize the policy
- [ ] Write the paper

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

### theoretic process

1. Agent $\Delta$ polls the environment for observations in time $t$
2. At each time $t$, $\Delta$ receives an observation $o_t\in O$
3. $\Delta$ uses policy $\pi$ to form and send an action $a_t\in A$
4. Each $o_t$ and $a_t$ form a sequence of (observation, action) tuples in a trajectory segment $\sigma$, where $\sigma=((o_0,a_0),(o_1,a_1),...,(o_{k-1},a_{k-1}))\in(O\times A)^k$
5. Whereas normally a reward function $r_t\in\Bbb R$ is applied, we have human indicate preference, as $\sigma^1\succ\sigma^2$, instead
6. Algorithm behavior evaluation can be either **quantitative**:
	Preferences $\succ$ are generated by a reward function $r:O\times A\rightarrow\Bbb R$ if:
	$((o_0^1,a_0^1),...,(o_{k-1}^1,a_{k-1}^1))\succ((o_0^2,a_0^2),...,(o_{k-1}^2,a_{k-1}^2))$
	whenever
	$r((o_0^1,a_0^1) + ... + (o_{k-1}^1,a_{k-1}^1)) > r((o_0^2,a_0^2),...,(o_{k-1}^2,a_{k-1}^2))$

7. Or **qualitative**:
	At each time $t$ the policy $\pi : O \rightarrow A$, and a reward function estimate $\hat{r}: O \times A \rightarrow\Bbb R$, each parameterized by deep neural networks

Questions?
Will the tool be concerned with timeliness? 
	eg. will the tool provide incorrect answers because they are out of date?

## implementation notes

### On RLHF and human preference

> "Of two acts $f$ and $g$, it is possible that the person **prefers $f$ to $g$**. Loosely speaking, this means that, if he were required to decide between $f$ and $g$, no other acts being available, he would decide on $f$. This procedure for testing preference is not entirely adequate, if only because it fails to take account of, or even define, the possibility that the person may not really have a preference between $f$ and $g$, regarding them as equivalent; in which case his choice of $f$ should not be regarded as significant." - Savage, The Foundations of Statistics, p.17

## references

[https://openai.com/blog/instruction-following/]
[https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/]
[https://github.com/openai/following-instructions-human-feedback]