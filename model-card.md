# ShakespeareGPT
---
Last updated: Jan 23

## Model Details
---
ShakespeareGPT is a GPT-style language model trained on the complete works of William Shakespeare. Using Reinforcement Learning through Human Feedback (RLHF) and an estimated parameter size $\sim 962k < x < \sim 5.3mn$. This methodology follows work done by OpenAI to produce a model that is finely-tuned for accuracy of responses (minimize untruthful answers) to imperative prompts.

### Model Date
Jan 23

### Model type
RLHF-tuned language model

### Paper & samples
[Proposal](../Shakespeare%20%20GPT%20paper.docx)

### Model version
0.1

## Model Use
---
Inexpert users will direct imperative statements ("Describe the character Henry V.") to the model, free from the burden of prompt-hacking. The model will be finely-tuned through an iterative process of RLHF and unsupervised learning to provide answers that pass an arbitrary truthfulness benchmark, and provide a confidence interval for each response.

## Data and Performance
---
### Data
The model is initialized from a [NanoGPT](https://github.com/karpathy/nanoGPT/blob/master/model.py) baseline, implemented with Model-View-Controller (MVC) architecture, in CPython. There will be three initial versions:

    1. A character-tokenized vocabulary.
    2. A sub-word-tokenized vocabulary.
    3. A word-tokenized vocabulary.

Each model will be tested for comparison against the other two. The benchmark will be answer coherence and truthfulness. An arbitrary number of labeling events will be chosen to train each model. Labelers will demonstrate desired model behavior by selecting a score per response, on an ordinal scale. The model will provide responses with an attached confidence, designed to be retrofitted to existing interval scales of confidence. The training data will be:

    1. Tiny Shakespeare
    2. The Complete Works of William Shakespeare, by the Gutenberg Project.
    3. Critical essays on the works of Shakespeare.

### Performance
This method and model was chosen due to the proclivity for human-feedback-tuned GPTs to return more truthful responses with fewer training parameters.

The MVC design allows this to scale into a future projects with much larger parameter sizes, which can run asynchronously to provide new inputs.

## Limitations
---
### Model Limitations
The model will be trained on a constrained data set, and as such, can only provide responses based on the data set, and the average critical analysis of Shakespeare.