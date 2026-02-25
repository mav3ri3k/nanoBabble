# Research Questions
These are the research questions I aim to answer through this work.

## Can PoLM be used to replicate ablation results from LLM
[Step 3.5 Flash](https://arxiv.org/pdf/2602.10604)
This model has a very interesting architecture of 3:1 ratio of Full Attention and SWA.
For this, they also share many ablations which were done on 30B-A3B model.

It would be very interesting and cost efficient to know if these same results
can be replicated at much smaller scale of 30m-250m using PoLM.

Therefore thing to find out:
Model performance on layouts:
1. FFFF
2. S1F1
3. S3F1
5. S3F1+Head
(Names taken from the paper)

### Limitations
While this might help identify model performance at smaller scale,
in case of the paper, they ended up using a slightly worse configuration
because it has better decode/prefill characterstics. Therefore likely better
perf/watt characteristics. This work would not be able to identify these things.

## Converting AR model to Diffusion models
[LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/pdf/2512.15745)
This model is interesting becaues it takes a pre-trained AR model and convert it to diffusion model.

It would be interesting to find out using PoLM techniques if the atomica capabilites
of the AR model hold up, or vanish after the conversion.

Therefore thing to find out:
Model performance on:
1. Diffusion pretrain
2. AR pretrain
3. AR -> Diffusion conversion
