An InstructGPT-based model to talk to the bard.

# question

How could a GPT provide accurate, intelligence quality answers to specific prompts by users based on a corpus of provided data?

## goal

To produce a GPT, trained on all shakespeare, that is queriable in plain English. We intend for this model to be able to answer an imperative natural language prompt such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt."

The goal of this research is  to provide a practical implementation of GPT3 that meets the desired end state of the customer.  The result of the research will be a GPT, trained on all Shakespeare's writing that will allow the user to ask plain text questions such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt." The query returned will be a timely, accurate, and truthful response that answers the intent of the users query.  Accurate answers are the paramount priority of the tool.  Answers to queries will  be returned with a relative probability  (between zero (0) and one (1) with one  being 100% accurate) that the returned answer is correct.  

## narrative 

        Fall to thy prayers. This statement is taken from the William Shakespeare
        play Henry the Fifth, in which King Henry disavows the knight Sir John Falstaff.
        Considered by itself, the statement to "Fall to thy prayers..." can be taken as an ominous
        directive implying some sort of threat. We could imagine a despot
        or tyrant telling his enemies to Fall to their prayers before he
        slaughters them. It inspires this imagery because of the encoding
        I apply to it based on my own experiences, preferences, and inclinations.
        However, when taken in the context of the play Henry the Fifth,
        we see Henry is essentially telling Falstaff to fall to his prayers
        because he has been disowned. He should fall to his prayers because
        we, the audience, know Falstaff to be an avid sinner. Henry is sealing
        Falstaff's fate and thus, his desolation. We derive this encoding from
        Shakespeare's body of work, including other plays. The body of Shakespeare's
        plays in which either Henry the Fifth or John Falstaff appear provides
        the context we require to understand Henry's directive.
        We expanded our context from the statement alone, to the statement in relation
        to Henry and Falstaff's other interactions of which we are aware.
        This is how human language is encoded. So, how does a computer which
        processes digital signals understand the difference between the encoding
        I applied to the statement, and the encoding Shakespeare intended.
        In fact, as an aside, how can we even be sure we completely understand
        the encoding Shakespeare intended? The answer is context.

        So, how would you, dear listener, seek to understand this statement?
        You would probably read the play, watch a stage performance, or read
        a critical essay. In fact, you'd be best served to do all three, thus
        expanding your context and building the encoding required to decode
        the intent behind this sentence. The computer is no different. We must
        find a way to allow the computer to understand this statement, but
        critically, in context with other statements. In fact, we can leverage
        the power of current computing to build a large contextual understanding.

        First, we have to transform the text which humans can read, into numbers
        which the computer can use. Humans use a complex and poorly-understood
        combination of logic, emotion, and memory to build context, and each of us
        applies our own gradient of these mental properties. What exactly that
        combination of properties that forms a mental state is varies from person
        to person. The computer, however, performs sets of instructions which
        usually involve either moving signals around between components, or doing
        calculations. So how does an inorganic machine gain context to interpret
        the encoding of human speech? Well, we only have one option: we have to
        compute the relationships between the components of human speech mathematically.

        First, we begin by exposing the invisible symbols the computer uses to make
        the text legible to us. For instance, the last two characters, a backslash and the
        letter "n", are essentially an instruction to the computer to start a new line when
        it encounters this combination. Now, let's isolate just one word, "Fall".

        Next, we transform the symbols that comprise human language into
        numbers, a process also called encoding. Interestingly, this version of encoding
        is slightly different from the encoding we described earlier, but you probably
        knew what I meant because of the context. For numbers, we use integers, which
        implies only zeros after the decimal place, known in math as whole numbers.
        Next, we choose unique integers to represent our letters. The numbers we choose
        can be arbitrary, or we can be clever and choose numbers which are attractive
        computationally, but let's ignore that for now. In this case, we've chosen the integers
        18, 39, and 50. Now that we've converted our letters to numbers, we can say they
        have a basic encoding. In order to carry these numbers through sets of computations,
        we place them into a computer data structure called an array, which can be thought of
        as a 1-dimensional matrix, a vector, or simply a named collection of values stored in
        contiguous computer memory. Now, let's place our encoded letters back into the context
        of the sentence, Fall to thy prayers. Note the last 4 values: 8, 8, 8, 0. We've also
        included the encoded version of a period, and the newline character. We do this
        because punctuation in human language is a form of encoding that describes how
        we should interpret the text. A comma might separate statements syntactically or
        signal a vocal pause if the text is meant to be read aloud.

        Now, let's view the original human-legible text, encoded into english letters.
        There's our word, "Fall", in the context of the play. Now, let's encode these three
        lines, including all the punctuation and hidden characters like newlines and spaces,
        into integers. If you were to deeply consider the integer-encoded text in this way
        you might realize that our original simple encoding won't unlock the computer's
        ability to interpret the speech because our encoding lacks spatial awareness.
        It lacks context. Each letter appears in other places, in other words.
        The letter "l" appears 7 times. The computer doesn't inherently understand the
        difference between the appearance of the letter "l" in the word "fall", to the
        letter "l" in the name Falstaff. Each of these letters, and indeed each instance
        of each letter must carry its own encoding.

## method

The first step will be to design a deep reinforcement learning with human feedback (RLHF) system, the aim of which is to design a reward function. That reward function will then be used to optimize to reward function of more traditional unsupervised reinforcement learning models.

The tool will utilize mixed methods to maximize the existing qualitative and quantitative algorithms to produce the best product.  The qualitative method will be used to provide human input to train the reward function which will assist in determining how the tool will respond to human inputs.  The resulting reward function will then be applied in an unsupervised quantitative way to train on the corpus of data, in this case the entirety of Shakespeare's written works.  

describe qualitative methods
describe quantitative methods
why did we use them together?

## roadmap
- [x] Select a GPT for our use case - [InstructGPT](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf)
- [ ] Research Reinforcement Learning from Human Feedback (RLHF)
- [ ] Theorize a practical application of the RLHF method for our use case
- [ ] Theorize python implementation of RLHF and practical use case
- [ ] Search for python implementations of InstructGPT and a similar RLHF method
- [ ] Reading list
- [ ] Let's code - NanoGPT (Andrej Karpathy)
- [ ] Select Reinforcement Learning algorithm [OpenAi baselines](https://github.com/openai/baselines), used to optimize the policy
- [ ] Write the paper
- [ ] Tokenization: begin with tiktoken BPE for sub-word encoding, test against sentence token encoding
- [ ] Select and test varying tokenized tensor block sizes for transformer context [explore memory limits][average the sentence length across a corpus]
- [ ] Identify specifications on our hardware (GPU parallel processing cores), and fine tune block and batch size
- [ ] Select optimizer (Karpathy recommends AdamW), tune learning rate
- [ ] Define our hyperparameters for compute-optimal training [DeepMind](https://arxiv.org/pdf/2203.15556.pdf)[Scaling Laws](https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb)

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

### Tokenized vocab lengths (complete works data set)
char tokenization = 107
tiktoken BPE = 50257
word tokenization = 76175

### On RLHF and human preference

> "Of two acts $f$ and $g$, it is possible that the person **prefers $f$ to $g$**. Loosely speaking, this means that, if he were required to decide between $f$ and $g$, no other acts being available, he would decide on $f$. This procedure for testing preference is not entirely adequate, if only because it fails to take account of, or even define, the possibility that the person may not really have a preference between $f$ and $g$, regarding them as equivalent; in which case his choice of $f$ should not be regarded as significant." - Savage, The Foundations of Statistics, p.17

## references

[Aligning Language Models to Follow Instructions](https://openai.com/blog/instruction-following/)

[Learning from Human Preferences](https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/)

[InstructGPT: Training Language Models to Follow Instructions with Human Feedback](https://github.com/openai/following-instructions-human-feedback), [paper](https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf)

[Baselines](https://github.com/openai/baselines/tree/master/baselines)

[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY), [nanoGPT repo](https://github.com/karpathy/nanoGPT), [lecture repo](https://github.com/karpathy/ng-video-lecture)

[The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)

[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

[Attention is All You Need](https://arxiv.org/abs/1706.03762)

[Training Compute_Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)

## development notes
- 25 Jan 23. adam. initializing the repository, architecural design, white paper framing.

	We've chosen the Model-View-Presenter as the base design in order to prepare this library for scaling to new datasets and implementations. The repository will come complete with a well-defined wiki, licensing, and other scale requirements. We want to try and future-proof the design from initialization. This design may change as the advantage becomes apparent.
	
- 7 Feb 23. adam. completing the model, optimizing for scale, hardware constraints.

	We've reached our hardware limit on personal machines without some very clever optimization. At writing there are our hyper parameters:
	
	    - device = 'cuda' if torch.cuda.is_available() else 'cpu'
	    - max_iterations = 20_000
	    - evaluation_interval = 500
	    - block_size = 256 # max context length
	    - batch_size = 64 # independent sequences in parallel
	    - trng_pct = .90
	    - learning_rate = 3e-4
	    - manual_seed = 7561
	    - embedding_table_dims = 32
	    - n_heads = 8
	    - n_layers = 8
	    - dropout = 0.2
	    
    	This was our generation:
	
	07-Feb-23 19:20:10 - INFO - Data decoded using char tokenization: 
	
		yess thou say all, dreavets woult that evar
		Dother?

		PRUMINA:
		It put, this alaster, I am my fow you?

		KING EDWARD IV:
		O matas of her, and Jeceral: you be will,
		Who him ilkli; our his friendmet, theseed, Which is Edcose;
		Instinor so a fit Batily hese; and Reeming hou'd
		Like not the my shallow, Wopphy a me yet
		I gate worry all gor of shalke us.

		Lesk RIVIARD I Sembalin:
		And by Verences, ro but speak her with event.
		Noop, thou, say with but too can I part lives!
		And it the flape then se stripow an...
	
