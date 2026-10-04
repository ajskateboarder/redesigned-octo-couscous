ai and human slop alert

`bilinear_attention_pr_experiment.py` implements the interesting dct variant

`natural_dct` experiments are just experiments that try training this dct on different token positions, like on mid-range token positions instead of last 3, and also to see if padding is mostly responsible for the method's various good facets. it turns out padding is *not* solely responsible (nor does it really matter but glad i checked), and mid-range tokens don't fundamentally change the type of interactions involved

these experiments also try applying the superellipse exponential fit from [this post](https://www.lesswrong.com/posts/uDrsffSLzWD6cDnTt/evidence-for-feature-specific-error-correction-in-llms-1) to DCTs trained over swiglu transformers. the goal here was mostly to try and transfer some relevant ontology here and see if it can explain things well or just fail. for various obvious-in-hindsight reasons the $p$ metric implications do not fully transfer over; $p \le 1$. superellipse fit creates two corner regions forming an AND-like interaction between the features (so $p > 2$ does not automatically mean privileged features per se)
this type of interaction mostly stands when adding $l,r$ features into either branch (either swiglu, or bilinear). it also maybe stands when just adding $l+r$ in one branch? not entirely sure, but that would be convenient

there's some interesting results to be described about what the corner regions describe, so the fit is possibly interesting and relevant for explaining, and differentiating dct factors. im gonna slowly distill this for a while most likely cus idk for sure