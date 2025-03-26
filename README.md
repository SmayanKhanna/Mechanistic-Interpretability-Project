# Investigating Lexical and Syntactic Complexity in Language Models via Mechanistic Interpretability 

`#Mechanistic Interpretability`, `#LLMs`

Smayan Khanna, Ian Zhang

# Abstract

Large Language Models have become increasingly prevalent in the modern world, but little is actually understood about how individual layers and blocks interact with each other and contribute to the generated outputs. The goal of our project is to shine some light on this black box, and attempt to understand where lexical richness and complexity develop within a GPT-2 model. We also examine whether certain “vectors” may encode complexity: can we add a series of values to activations which output simple text and consequently increase output complexity (and vice versa)? Our experiments suggest that linguistic complexity is shaped by specific activation layers, though the effects vary significantly depending on the structure and content of the prompt. We also find that complexity tends to build up in the deeper layers of the model and behaves directionally, with the effects of adding or removing complexity depending on which way the patching is applied.

# What this project is about

The internal mechanisms that give rise to linguistic richness and complexity remain poorly understood in Large Language Models. In particular, it is unclear how different layers of a transformer encode and manipulate lexical richness (word variety and complexity) and syntactic complexity (sentence structure). Our project aims to shed light on these internal processes using **mechanistic interpretability (MI)** techniques, specifically **activation patching** - a method where we swap intermediate activations between different inputs to identify which layers or neurons contribute to specific model behaviors.

While existing work in interpretability has identified circuits responsible for individual tasks (e.g., factual recall, syntax parsing), little attention has been given to how models encode **linguistic complexity** itself. Do deeper layers contribute more to complex language formation? Can we intervene in a model's processing to deliberately make its outputs simpler or more sophisticated? 

To explore this, we analyze GPT-2’s internal activations to determine how "complexity" develops across layers. We use two datasets: **BabyLM**, which consists of child-directed speech, and **Wikipedia**, which represents formal adult text. Both are sourced from HuggingFace and were chosen because they naturally provide a clear contrast in complexity.

**Broadly, our methodology consists of two key experiments:**

1. **Activation Patching Experiment:** We swap activations between simple and complex text inputs at different layers of GPT-2 to determine where linguistic complexity is encoded. 

2. **Vector Addition Experiment:** We test whether a "complexity vector" can be identified and used to artificially modify text complexity. By adding or subtracting learned activation patterns, we investigate whether complexity is encoded in a structured, manipulable way.

# Approach

**Fine-tuning & Complexity Controlling**

We fine-tune a pre-trained GPT-2 model (114M parameters) to explicitly control linguistic complexity. Following the controlled text generation approach used in the CTRL model (Keskar et al., 2019), we introduce two control tokens—[BABY] for simpler text and [ADULT] for more complex text. Each sentence from the BabyLM dataset is prepended with the [BABY] token, while sentences from the Wikipedia corpus are prepended with the [ADULT] token. This enables the model to adjust its output complexity based on the control token - allowing us to directly manipulate language complexity through prompting.
To ensure balanced training, we use approximately 1.5 million tokens from each dataset and structure fine-tuning in two sequential phases—first training on the [BABY] sentences, followed by the [ADULT] sentences. We fine-tune the model for 3 epochs with a batch size of 8 on Google Colab using A100 GPUs. 

## **Experimental Methodology:**

To the best of our knowledge, our specific experimental methodology for investigating where complexity originates in LLMs is novel. 

However, we build upon substantial research and work of others to accomplish this project. We heavily utilize the [TransformerLens library](https://transformerlensorg.github.io/TransformerLens/) (Nanda et al., 2022) to perform our activation observation and patching tasks. We decide to prefix simple and complex language examples with corresponding [Baby] and [Adult] tokens with guidance from the CTRL model (Keskar et al., 2019). Our vector patching experiment takes inspiration from a (Arditi et al., 2024), which utilizes vector patching to bypass model refusal “filters” for ill-willed requests. Additionally, we draw from key mechanistic interpretability studies. The ROME paper (Meng et al., 2022) explores patching mid-layer MLPs to alter model knowledge while preserving coherence, and the IOI paper (Wang et al., 2022) analyzes how GPT-2 identifies indirect objects. We also take insights from Dumas et al. (2025), which examines how transformers process language-agnostic concepts via activation patching. Finally, we structure our approach in accordance with Neel Nanda’s research on the best practices for Mechanistic Interpretability (Nanda et. al 2023).

### Experiment 1: Activation Patching 

In Experiment 1, we perform activation patching by systematically swapping activations across different layers within GPT-2’s transformer architecture using TransformerLens, a state-of-the-art library for mechanistic interpretability.
Our goal is to evaluate how injecting activations from a forward pass of our fine-tuned model on an "adult" prompt(e.g., "[ADULT] Dogs like to play!") affects the per-token prediction complexity of an equivalent "baby" prompt(e.g., "[BABY] Dogs like to play!")—and vice versa.

**How does activation patching work?**

Activation patching (also known as causal tracing) is a technique for analyzing which parts of a model contribute to specific behaviors. Instead of modifying the model’s weights, we intervene at inference time by replacing activations (intermediate representations) from one input with activations from another and observing the effect on the output. To ensure a fair comparison, the prompts remain identical except for the control token at the beginning of the sentence. This allows us to isolate the effect of activation patching and determine whether it truly influences the complexity of the output.

The diagram below illustrates how activation patching works and how we apply it in our experiment:

<img width="464" alt="image" src="https://github.com/user-attachments/assets/1d56ad30-6db1-4e46-832f-8adab47431c5" />

1.Baseline Forward Passes:
> We first run two inputs (e.g., "[ADULT] Dogs like to" and "[BABY] Dogs like to") through the model separately.
> We store all activations (which are the outputs of each of the many individual layers in the model) from these runs without modifying the model. 
> We use the output of the runs to calculate our baseline complexity scores:
> AC = Complexity score for the adult prompt.
> BC = Complexity score for the baby prompt.
2. Activation Patching:
> We select a specific activation (e.g., from a particular layer and component - say for example the out of MLP in the 5th transformer block) from the adult prompt.
> We inject this stored activation into the forward pass of the baby prompt.
> The model then completes inference with this modified activation and gives us a new output from which we calculate a modified Complexity score PC.
3. Measure the Effect:
> We compute the impact of the patched activation using the formula: $\text{Impact Score} = \frac{PC - BC}{AC - BC}$
> This score helps us assess the impact of the patching. It's rescaled so that the adult baseline is 1 and baby baseline is 0. That means say that for example the score is 1.2, patching increased the complexity of the baby output _even more_ than the adult baseline.

Now, we essentially repeat this process many times, patching the output for each layer and token output in the transformer model.

The diagram and explanation above explain how we take the activations of an [ADULT] forward-pass and patch them into a [BABY] forward-pass and measure the difference in complexity. We call this the [ADULT] → [BABY] Patch. 

We also need to do the opposite: [BABY] → [ADULT] Patch. The procedure of patching activations of a [BABY] forward pass into a [ADULT] forward pass is essentially the same, except we change how we we compute the impact of the patched activation. For the [BABY] → [ADULT] Patch, we use the formula: $\text{Impact Score} = \frac{PC - AC}{AC - BC}$. 
Now, the score is rescaled so that the adult baseline is 0 and baby baseline is -1. 

To summarize:

1) [ADULT] → [BABY] Patch

The [BABY] baseline = 0 (simpler language).

The [ADULT] baseline = 1 (higher complexity).

2) [BABY] → [ADULT] Patch

The [BABY] baseline = -1 (simpler language).

The [ADULT] baseline = 0 (higher complexity).

We hypothesized that patching baby activations with adult activations would increase the complexity score predicted by this metric while the opposite would happen if we patched the reverse. 


**How do we measure shifts in complexity?**

When we perform a forward pass of a prompt, such as "[ADULT] Dogs like", we obtain the logits associated with each token in the sequence. These logits represent the model’s predicted probability distribution over possible next tokens. For example:
Given the token "Dogs", the model might assign the highest probability to words like "are" or "like" etc.
Given the token "like", (and the previous tokens in the prompt) the model might then predict words like "cheese" or "socializing" as the most likely continuations.

To quantify changes in complexity, we compare the top 10 predicted logits for each token in the sequence across both prompts—one using the [ADULT] control token and one using the [BABY] control token. Importantly, we ignore the logits associated with the control tokens themselves, as they are only used for conditioning and do not contribute directly to content generation.

**Complexity Metric**

To compare the complexity difference between two prompts (e.g., "[ADULT] Dogs like" and "[BABY] Dogs like"), we analyze the predicted token distributions using a frequency-based complexity score:

1. We retrieve the top 10 candidate completions for each token in the sequence based on model logits.
2. We filter out punctuation, white-spaces or special characters like “\n” (new-line)
3. For the valid words out of the top 10 candidate words, we compute a frequency score $C(w) = -\log_{10} \left( f(w) + 10^{-9} \right)$,. The frequency value $f(w)$ is computed using the word frequency python library, which estimates how rare a word is based on its prevalence in google books, wikipedia etc. Lower frequency/ rarer words get a higher “complexity score”.
4. We average this score across the candidate words for each token 
5. We then aggregate these scores the prompt  and take its average to get a final complexity value  

We find that including the [ADULT] tag consistently increases the complexity value as compared to the [BABY] tag for the same prompt. With this, we get a quantitative way to compare the complexity of two identical prompts with different control tokens.

Word frequency has been widely used as a proxy for text complexity in educational research and NLP applications [[1]](#references). However, it should be mentioned that this method gives us an approximate heuristic for modeling the complexity of the logits for all input tokens. We **are not** decoding output sentences in this experiment: activation patching is a method that can be used to sensitively probe the internals of the model, but the complexity score calculated in experiment 1 is not a definitive measure for linguistic complexity.

In fact, decoding coherent long-form outputs while applying activation patching remains a highly technical, open problem in mechanistic interpretability research—well beyond the scope of this project.

### Experiment 2: Vector Patching 

In **Experiment 2**, we test the hypothesis that activations for simple language are consistently and distinctively different from those for complex language. We first estimate a "complexity difference vector" by computing the activation difference between complex ([ADULT]) and simple ([BABY]) prompts. We then add this vector to activations from simple prompts to see if it systematically increases the complexity of the generated outputs.

<img width="408" alt="image" src="https://github.com/user-attachments/assets/e7c43c09-6dfa-4365-bfd1-eea663a32950" />

Specifically, we first take a random sample of 1000 Baby training examples, and 1000 Adult training examples, performing forward passes for each prompt. During the forward pass, we used TransformerLens to track the activations of every neuron across each layer. We then compute the average difference between neuron activations across Baby- and Adult-labeled prompts, for each layer of our model. We hypothesize that this “vector” may capture some data relevant to lexical richness of output. Specifically for this project, we are interested in the Residual Flow, Attention Head Output, and MLP Outputs, but similar methodology can be applied to other parts of the GPT2 model.

<img width="733" alt="image" src="https://github.com/user-attachments/assets/a8178da2-fe8f-4015-ad1b-df293b93c7eb" />

To patch, we randomly choose 30 [Baby] prompts from our dataset, which we run forward passes on. During these forward passes, we utilize TransformerLens to edit the activations, “adding back” the previously calculated vector, before letting the model finish the forward pass and generate 50 tokens of text. We repeat this process to obtain a result for each of the 12 layers of our GPT-2 model for each of the internals of interest (Residual Stream output, Attention Head output, and MLP output). We then estimate lexical richness of the resulting generations, using metrics such as TTR (type-token ratio) and MTLD (measure of textual lexical diversity) from the LexicalRichness library. We finally compute these metrics using the same prompts run on a completely un-patched model as a basis for comparison.

<img width="653" alt="image" src="https://github.com/user-attachments/assets/50c18fc4-1863-4e55-9a28-4c95e661521a" />

[Image Source](https://medium.com/@vipul.koti333/from-theory-to-code-step-by-step-implementation-and-code-breakdown-of-gpt-2-model-7bde8d5cecda)

# Results

## **Experiment 1:**

As we explained earlier, we both patch a [BABY] forward pass with adult activations and patch and vice versa. We conduct a series of tests based on the methodology explained above. First, we test this on three different topics, selecting two prompts of different lengths for each topic to check whether sentence length affects the results. Then, we select two prompts of similar length which encode the same meaning but are written with vastly different levels of language and vocabulary.

We assume the reader is familiar with how transformer blocks and multi-headed attention work. GPT-2’s architecture  is composed of 12 stacked transformer blocks. Each layer consists of multi-head self-attention layer followed by an MLP, with residual connections and layer normalization applied throughout.

We patch three types of activations:
1. Pre-Residual Stream – The residual stream before each transformer block.
2. Attention Head Output – Each attention head across all layers.
3. MLP Output – The output of the feed-forward network in each transformer block.

**How to interpret plots?**

If the reader is unfamiliar with Mechanistic Interpretability studies, these plots might be confusing at first. Here's a quick breakdown of how to interpret them (feel free to skip this section if familiar):

![image](https://github.com/user-attachments/assets/b23dd171-b5f4-420c-8be6-2f3f2278051a)

The heatmaps show how much replacing baby activations with adult activations changes the model's behavior, measured by our complexity score.

Red values = the patch at layer _X_ increased the complexity score for token _Y_. If the value is above 1, it's more complex than the adult baseline.

Blue values = the patch at layer _X_ decreased the complexity score for token _Y_. If the value is below 0, it's more less complex than the baby baseline.

Gray/white areas = little to no change.

For components like the residual stream and MLP outputs, activations exist separately for each token in the prompt at each layer. This means we can patch the activations for just one token at a specific layer. For attention head outputs, patching works differently. Here, we replace the activations for entire attention heads at specific layers. 

Keep in mind that if we measure replacing adult activations with baby activations, the baselines change and we explained this earlier.

**Topic 1: Politics**

We choose two prompts: 

“The election is close” and “The election is close, the country is on edge”. We prepend the control tokens as usual and get the following results:

![Unknown-2](https://github.com/user-attachments/assets/eab67e94-890c-491c-9b50-0aee127b71aa)

![Unknown](https://github.com/user-attachments/assets/949c43d6-55ed-497e-987f-1478d3cbe5c9)

Patching the baby forward-pass with adult activations results in a marginal increase in complexity across most attention heads for both the short and long versions of the prompt. The changes are relatively scattered, with no single head or layer dominating the effect.

For the residual stream, we observe that patching only starts to have an impact after the first few layers, suggesting that complexity-related information may accumulate later in the model. However, the overall effect remains subtle, with some tokens showing slight increases and others slight decreases in complexity.

Similarly, patching the MLP output shows only small, scattered increases in complexity across layers and tokens, without strong, consistent patterns across the prompt lengths.

![Unknown-3](https://github.com/user-attachments/assets/4a67260c-ba82-4c27-b14b-478f30ac995d)

![Unknown-4](https://github.com/user-attachments/assets/c5325e20-7ca1-42ce-b35f-3a8aa5b09cbc)

Patching the adult forward-pass with baby attention heads results in a marginal increase in complexity across most heads, which is slightly contradictory to what we might expect. Instead of decreasing complexity, the changes are scattered and inconsistent across the attention heatmap.

We see a similar pattern for the longer prompt. For both the residual stream and MLP outputs, certain tokens—like election—show a noticeable spike in complexity.

As with the previous cases, patching the residual stream mainly has an effect in the later layers, suggesting that complexity is being built up deeper in the model. However, even there, the effects are mixed, with some layers and tokens increasing in complexity and others decreasing.

**Topic 2: Simple sentences**

We choose two prompts: 
“Dogs like to play” and “Dogs like to play! They are very friendly”. 

![Unknown-5](https://github.com/user-attachments/assets/31d98bb0-0d46-4c39-9bd2-77d35457a4f8)

![Unknown-6](https://github.com/user-attachments/assets/42edc73c-39a2-4dc8-95f8-ab31dc41bcfd)

These results are strange - the "dogs" token sees a massive decrease in complexity when we inject adult activations for both the residual stream and MLP output for both prompts. However patching attention heads has a net increase in complexity for the shorter prompt. For the longer prompt, injecting baby activation increases the complexity score for certain words and many attention heads. It's hard to ascertain any clear pattern from these heatmaps except that complexity shifts for the MLP and residual stream outputs typically happen deeper in the model. 

![Unknown-7](https://github.com/user-attachments/assets/98068eba-670b-4965-92ce-22584924410e)

![Unknown-8](https://github.com/user-attachments/assets/e74bbc71-de8c-4958-9d77-6b6f00eb5efc)

Patching attention heads has a variable effect on complexity, with no clear, consistent pattern. We also see that patching later layers generally has a stronger effect, once again suggesting that complexity is encoded deeper in the model. As we see, when patching baby activations into the adult forward-pass, complexity sometimes increases instead of decreasing which is an unexpected result

**Topic 3: Science**

We choose two prompts: 
“Photosynthesis occurs when” and “Photosynthesis is a highly complex process that occurs when”. 

![Unknown-9](https://github.com/user-attachments/assets/4c3f1f8a-e823-4406-bb9d-1de023b88b15)

![Unknown-10](https://github.com/user-attachments/assets/f7c85f3f-5b00-4fb4-a599-eef6ed9e653a)

This finding is particularly strange. For the shorter prompt, patching baby activations with adult activations leads to a net increase in complexity, which is exactly what we would expect. However, for the longer prompt, we see the opposite: patching baby activations with adult activations actually decreases the complexity score. This is unexpected and suggests that the length or structure of the prompt itself might be interacting with the patched activations in unpredictable ways.

One possible explanation is that in longer prompts, the model may distribute complexity differently across tokens, and the added context may shift how much influence the patched activations have.

![Unknown-11](https://github.com/user-attachments/assets/e297c6da-02b8-458e-91e3-87e6120993d2)

![Unknown-12](https://github.com/user-attachments/assets/de6fbea3-8bc1-4f27-85c9-96976ebf770a)

We see a general increase in complexity when patching adult activations with baby activations for the longer prompt. This is, again, the opposite of what we expect. The shorter prompt behaves more in-line with expectations, with a decrease in complexity observed for many attention heads. 

**Test: Sentences of varying complexity that encode the same meaning**

"The dog is happy." is our simple, common sentence, while "The seraphic hound exudes euphoria." uses rarer vocabulary and more elaborate phrasing to say the same thing. Like usual, we prepend the control adult and baby tokens. 

![Unknown-13](https://github.com/user-attachments/assets/027e9a68-7a3a-42d9-91c6-963294e2b707)

For the simple prompt, patching baby activations with adult activations leads to a net increase in complexity, which is exactly what we would expect. We also clearly see how patching activations deeper in the model has a stronger effect in changing the complexity. 

![Unknown-14](https://github.com/user-attachments/assets/165e183e-9cc4-4b31-ad2e-cc949cd1a7d7)

For the complex prompt, patching baby activations with adult activations leads to a varied change in complexity. Based on the results we've seen so far, it seems like our complexity metric is much more sensitive to the length of the input prompt rather than the specific topic of the prompt. That's okay, because our goal isn’t necessarily to capture topic-level complexity, but rather to probe how structural differences—like prompt length or token-level composition—affect internal model behavior.

![Unknown-15](https://github.com/user-attachments/assets/fee6b237-1626-4285-b166-40f9d30ed35a)

![Unknown-16](https://github.com/user-attachments/assets/153d9e53-8455-46b7-af81-aefaec0c16b6)

When we do the opposite—patching adult activations with baby activations—we get exactly what we expected: a net decrease in complexity across all layers for both prompts. This highlights the fundamental asymmetry of activation patching—complexity is likely **directional**.

## Experiment 2: Vector Addition Patching

### Patching Residual Stream (BABY -> ADULT)

The results for Baby-to-Adult patching on the residual stream outputs are shown below. Each row shows lexical richness measures of a 50-token generation after the shown layer is patched. The first column denotes TTR (Type-Token Ratio), and the second column denotes MTLD (Measure of Textual Lexical Diversity). The chart also shows our unpatched baseline, compared to our patched experimental results.

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-bbc80305-7fff-58c3-a8c1-4091df3e6580">
Layer Patched | TTR | MTLD
-- | -- | --
**Baseline** | **0.2907** | **10.6866**
0 | 0.3039 | 11.4099
1 | 0.3006 | 11.3766
2 | 0.2766 | 11.2115
3 | 0.2865 | 11.4406
4 | 0.2882 | 9.7863
5 | 0.3121 | 11.6086
6 | 0.3101 | 11.8045
7 | 0.3157 | 11.5300
8 | 0.2928 | 11.3082
9 | 0.3049 | 11.5821
10 | 0.3334 | 13.7128
11 | 0.3141 | 11.7736

</b>

In the chart below, we can see that the lexical richness measures of the unpatched baseline baby prompts (orange) are usually lower than our patched outputs (blue). We can also see that patching in the first few layers has relatively little impact upon the lexical richness of output generated. Patching in layers 4-6 began to have a noticeable effect upon the output richness, but the largest differences were noticed in the latest layers, of 10-12. 

<img width="475" alt="image" src="https://github.com/user-attachments/assets/a82e81f8-5fec-45eb-aa22-eb9611b3773c" />

### Patching Residual Stream (ADULT -> BABY)

We then do the same thing, patching prompts starting with [Adult], and attempting to lower lexical richness of generation by subtracting the calculated vector during each forward pass. The results of Adult-to-Baby patching are shown below. Across our relatively small sample set (of 30 prompts), our TTR of patched generations is overall lower than the TTR of unpatched generations, showing that our “backwards” patching likely had some effect in decreasing the complexity of output.

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-50da332c-7fff-d65b-8d8e-75e7a3b55f75">
Layer Patched | TTR | MTLD  
-- | -- | --  
**Baseline** | **0.3851** | **14.8550**
0 | 0.3849 | 15.4358  
1 | 0.4000 | 14.2381  
2 | 0.3926 | 16.2939  
3 | 0.3736 | 13.7233  
4 | 0.3610 | 14.2379  
5 | 0.3370 | 12.3633  
6 | 0.3408 | 13.7735  
7 | 0.4069 | 16.8544  
8 | 0.3874 | 13.7172  
9 | 0.4033 | 13.7560  
10 | 0.3806 | 11.5472  
11 | 0.3955 | 11.9291  
</b>

However overall, our results are not as distinctive as our results from the Baby-to-Adult patching. Although on average, TTR and MTLD decreased, there are certain layers in which the TTR and MTLD of generated output actually increased after patching. This might be because of our relatively small number of prompts, along with the random choice of prompts to begin with. Patching the central layers (3-6) with the Adult-to-Baby vector resulted in significantly lower TTR and MTLD. Patching earlier layers did not have much of an effect on either TTR or MTLD, and in some cases even increased these measures. Finally, patching later layers had a mixed effect: TTR slightly increased, whereas MTLD decreased with the addition of the Adult-to-Baby vector.

<img width="481" alt="image" src="https://github.com/user-attachments/assets/65ab0e62-e2e7-4951-a01d-91b30b1b31b9" />



### Patching Attention Head (BABY -> ADULT)

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-50da332c-7fff-d65b-8d8e-75e7a3b55f75">
Layer Patched | TTR | MTLD
-- | -- | --
**Baseline** | **0.3125** | **11.4140**
0 | 0.3117 | 11.8888
1 | 0.3379 | 12.5244
2 | 0.3114 | 11.4846
3 | 0.3479 | 11.8850
4 | 0.3347 | 13.3861
5 | 0.3071 | 12.5175
6 | 0.3044 | 11.9072
7 | 0.3163 | 11.1162
8 | 0.3262 | 13.5206
9 | 0.3152 | 13.1585
10 | 0.3312 | 11.6986
11 | 0.3088 | 12.6770
</b>

<img width="657" alt="image" src="https://github.com/user-attachments/assets/d05640db-4de5-496d-941b-1dfec1b8bd63" />

We repeat the above experiment, except with the attention head outputs. The results we observe show that on average, adding the Baby-to-Adult vector to our [Baby] activations increased the lexical complexity, regardless of the layer in which we performed the addition. However, we do observe some "spikiness" in the results, such that the output lexical complexity varied significantly, depending on which layer was patched. This resulted in the observation that on certain attention heads, adding the Baby-to-Adult vector actually decreased the generated output's TTR and MTLD measures by a small margin. We believe this is because of our small number (30) of random starting [Baby] prompts, and thus a larger sample of starting prompts may smooth out these numbers.



### Patching Attention Head (ADULT -> BABY)

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-50da332c-7fff-d65b-8d8e-75e7a3b55f75">
Layer Patched | TTR | MTLD  
-- | -- | --  
**Baseline** | **0.4811** | **17.5728**
0 | 0.4795 | 16.1952  
1 | 0.4589 | 15.8409  
2 | 0.4303 | 15.2643  
3 | 0.4662 | 16.1960  
4 | 0.4302 | 15.2831  
5 | 0.4651 | 16.8884  
6 | 0.4233 | 18.7312  
7 | 0.4600 | 19.2714  
8 | 0.4539 | 16.9073  
9 | 0.4659 | 18.1911  
10 | 0.4451 | 14.9747  
11 | 0.4505 | 14.4340  
</b>

<img width="687" alt="image" src="https://github.com/user-attachments/assets/f6c267c2-cc07-4403-a4a1-2beace683676" />

Starting with [Adult] prompts and performing patching in the opposite direction, we observe similar results: There is still "spikiness" in the data, but we can see clearly that TTR has decreased across the board. MTLD also decreased on average, although not without the observation that in layers 6-8, MTLD actually increased after patching.



### Patching MLP Output (BABY -> ADULT)

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-4ea6b2e6-7fff-4ef3-47fb-49299b6508a5">
Layer Patched | TTR | MTLD
-- | -- | --
**Baseline** | **0.3288** | **13.2616**
0 | 0.3252 | 13.1580
1 | 0.2985 | 11.9611
2 | 0.3195 | 12.1673
3 | 0.3201 | 11.7551
4 | 0.3286 | 12.8736
5 | 0.3216 | 12.0752
6 | 0.3229 | 13.4047
7 | 0.3257 | 12.5235
8 | 0.3228 | 13.0052
9 | 0.3317 | 12.8009
10 | 0.3089 | 12.1896
11 | 0.3276 | 11.7283
</b>

<img width="698" alt="image" src="https://github.com/user-attachments/assets/ffd1574f-a3f1-4d18-a757-503d622ba30b" />

We finally repeat the experiments on the MLP Output of the transformer. Here, our results are quite striking: Despite patching the [Baby] prompts by adding the MLP Baby-to-Adult vector, we actually see a decrease in lexical richness and MTLD, pretty much across the board. This is very surprising to see such a significant, counterintuitive trend. And this trend actually continues into the Adult-to-Baby patching experiment:

### Patching MLP Output (ADULT -> BABY)

<meta charset="utf-8"><b style="font-weight:normal;" id="docs-internal-guid-50da332c-7fff-d65b-8d8e-75e7a3b55f75">
Layer Patched | TTR | MTLD  
-- | -- | --  
**Baselin** | **0.4692** | **17.3938**
0 | 0.4777 | 16.8227  
1 | 0.5148 | 18.1991  
2 | 0.4766 | 17.5105  
3 | 0.4820 | 16.1629  
4 | 0.4810 | 16.3332  
5 | 0.4540 | 14.7675  
6 | 0.4949 | 16.5400  
7 | 0.4960 | 17.4675  
8 | 0.4766 | 16.1052  
9 | 0.4842 | 16.7660  
10 | 0.4962 | 17.7004  
11 | 0.5095 | 20.7556  
</b>

<img width="680" alt="image" src="https://github.com/user-attachments/assets/0209dfc4-7180-424a-9daf-ada8f78d0aae" />

Again, counterintuitively, when we patched activations from prompts labeled [Adult] by adding the Adult-to-Baby vector, we actually observed an _increase_ in complexity, whereas we would expect a decrease: TTR was quite high across the board, while MTLD varied a bit more. We are not exactly sure why this phenomenon occurred, but we hypothesize that modifying the MLP Outputs causes interference with the residual flow component of the GPT2 transformer when the two sources are added and normalized, which may jumble the results or modify them in unpredictable ways. This may also be a simple consequence of a very unlucky random choice of starting prompts. However, future work will need to be done to replicate these results, and to explain the reasoning behind this phenomenon.


## Discussion

**Experiment 1:**

The results of Experiment 1 reveal several insights. First, complexity appears to be encoded more strongly in deeper layers, particularly within the residual stream and MLP outputs. Patching early layers tends to have little effect, while later layers produce more noticeable changes.

Second, complexity encoding is highly prompt-dependent. Given that our model was fine-tuned on a very small (and biased) dataset, the way complexity manifests can vary significantly depending on the structure of the input prompt.

Third, changes in early token complexity do not always reflect overall sentence complexity. This is especially relevant for longer, more structured sentences, where complexity may build gradually across the sequence. Adult prompts, for example, tend to produce more deliberate, structured outputs, whereas baby prompts introduce simpler relationships between words earlier on.

Finally, complexity is **directional**: patching baby activations with adult activations tends to increase complexity, while patching adult activations with baby activations decreases it. However, these effects are not always symmetric or uniform—depending on the prompt length and structure, the impact of patching can vary widely. This suggests that linguistic complexity isn't just a static property of individual layers or tokens, but something that builds and shifts across the model's computation in a way that depends on both context and direction of intervention.

Experiment 1 highlights the challenge of defining complexity itself. Since we are not analyzing full decoded outputs, but instead computing complexity scores from the top-10 logits of each token, our results offer a glimpse into the model’s internal representations rather than a direct measure of sentence complexity. We addressed these shortcomings in an earlier section. 

**Experiment 2:**

The results of Experiment 2 are also quite insightful:

First, we can see from our preliminary results that GPT2 neuron activations that generate text with lower lexical richness are indeed different from GPT2 neuron activations that generate text with higher lexical richness. And by computing the average difference in activations, we are able to "hack" GPT2 into generating more-or less-complex text than in its original state by adding or subtracting a previously calculated "complexity vector", containing the difference in activations from simple prompts and complex prompts.

Second, the lexical richness and complexity of output, along with the effectiveness of patching activations by adding or subtracting the "complexity vector" is very highly prompt- and layer-dependent. As a result of randomly sampling prompts as inputs to our experiments, we observed outputs that are visually very "spiky", and slightly changing the layer of GPT2 that we patched often resulted in large differences in output complexity.

Finally, we observed that while our results are relatively consistent for the GPT2 model's residual flow and attention heads, the MLP outputs seem special, especially since we observed an _opposite_ effect as expected: Adding the Baby-to-Adult vector to [Baby] activations actually decreased output complexity, and adding the Adult-to-Baby vector to [Adult] activation increased complexity. Again as stated above, we hypothesize this may be because of interference between the MLP and residual flow, but more work will be needed to investigate this phenomenon.

**Future directions:**

Together, Experiment 1 and Experiment 2 give us both a fine and coarse-grained view of how LLMs like GPT-2 encode complexity in language. We hope that our dual approach allowed us to identify significant patterns in layer representation.

To improve our project, and for future study we believe we can do the following:
1. Firstly, we think that extrapolating these techniques to larger models like GPT-2 large or potentially Qwen or even Llama 3.1 by Meta would be insightful. Perhaps more complex models encode these representations differently or more clearly. Currently, we were not able to do that due to compute limitations.
2. For the fine-grained experiment (Experiment 1), we would like to develop a more robust complexity metric for the results of activation patching. Decoding entire sentence generations after patching individual activations remains a challenging open problem; however, we feel that it would make our experimentation more robust and concrete.
3. For the coarse-grained experiment (Experiment 2), we would like to perform further investigation into the results from patching the MLP Output, as it seems to give us unintuitive results: we would like to examine exactly why we achieved the results we did. Further, we would also like to experiment with different decoding methods, and see if that affects our results at all.

(The content is based on Stanford CS224N’s Custom Final Project.)

## References
1. **Neel Nanda (2023).** *Mechanistic Interpretability* (https://www.neelnanda.io/mechanistic-interpretability)
2. TextProject. *Teaching Complex Text: Why Look at Word Frequency?* (https://textproject.org/teaching-complex-text-why-look-at-word-frequency-d35/)
3. **Wang et al.** *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small* (https://arxiv.org/pdf/2211.00593)
4. **Meng et al.** *Locating and Editing Factual Associations in GPT* (https://arxiv.org/pdf/2202.05262)
5. **Fred Zhang, Neel Nanda** *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (https://arxiv.org/pdf/2309.16042)
6. **Keshar et al.** *CTRL: A Conditional Transfomer Language Model for Controllable Generation* (https://arxiv.org/pdf/1909.05858)
7. **Arditi et al.** *Refusal in Language Models Is Mediated by a Single Direction* (https://arxiv.org/abs/2406.11717)
