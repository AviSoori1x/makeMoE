# makeMoE

<div align="center">
    <img src="images/makemoelogo.png" width="500"/>
</div>


<a href="https://www.databricks.com/product/machine-learning">
    <img src="https://raw.githubusercontent.com/AviSoori1x/makeMoE/main/images/databricks.png" width="50px" height="auto">
</a>
<br>
<span>Developed using Databricks with ❤️</span>



#### Sparse mixture of experts language model from scratch inspired by (and largely based on) Andrej Karpathy's makemore (https://github.com/karpathy/makemore) :)

HuggingFace Community Blog that walks through this: https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch

Part #2 detailing expert capacity: https://huggingface.co/blog/AviSoori1x/makemoe2

This is an implementation of a sparse mixture of experts language model from scratch. This is inspired by and largely based on Andrej Karpathy's project 'makemore' and borrows the re-usable components from that implementation. Just like makemore, makeMoE is also an autoregressive character-level language model but uses the aforementioned sparse mixture of experts architecture. 

Just like makemore, pytorch is the only requirement (so I hope the from scratch claim is justified).

Significant Changes from the makemore architecture

- Sparse mixture of experts instead of the solitary feed forward neural net. 
- Top-k gating and noisy top-k gating implementations.
- initialization - Kaiming He initialization used here but the point of this notebook is to be hackable so you can swap in Xavier Glorot etc. and take it for a spin.
- Expert Capacity -- most recent update (03/18/2024)

Unchanged from makemore
- The dataset, preprocessing (tokenization), and the language modeling task Andrej chose originally - generate Shakespeare-like text
- Causal self attention implementation 
- Training loop
- Inference logic

Publications heavily referenced for this implementation: 
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-Of-Experts layer: https://arxiv.org/pdf/1701.06538.pdf
- Mixtral of experts: https://arxiv.org/pdf/2401.04088.pdf

makeMoE.py is the entirety of the implementation in a single file of pytorch.

makMoE_from_Scratch.ipynb walks through the intuition for the entire model architecture and how everything comes together. I recommend starting here.

makeMoE_from_Scratch_with_Expert_Capacity.ipynb just builds on the above walkthrough and adds expert capacity for more efficient training.

makeMoE_Concise.ipynb is the consolidated hackable implementation that I encourage you to hack, understand, improve and make your own

**The code was entirely developed on Databricks using a single A100 for compute. If you're running this on Databricks, you can scale this on an arbitrarily large GPU cluster with no issues, on the cloud provider of your choice.**

**I chose to use MLFlow (which comes pre-installed in Databricks. It's fully open source and you can pip install easily elsewhere) as I find it helpful to track and log all the metrics necessary. This is entirely optional but encouraged.**

**Please note that the implementation emphasizes readability and hackability vs. performance, so there are many ways in which you could improve this. Please try and let me know!**

Hope you find this useful. Happy hacking!!
