# shakespeare
An InstructGPT-based model to talk to the bard

## goal

To produce a GPT, trained on all shakespeare, that is queriable in plain English. We intend for this model to be able to answer an imperative natural language prompt such as "Describe the character Henry the Fifth.", with a coherent answer such as "Henry the Fifth was King of England and fought in the Battle of Agincourt."

## method

The first step will be to design a deep reinforcement learning with human feedback (RLHF) system, the aim of which is to design a reward function. That reward function will then be used to optimize to reward function of more traditional unsupervised reinforcement learning models.


## references
[https://openai.com/blog/instruction-following/]
[https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/]
[https://github.com/openai/following-instructions-human-feedback]
