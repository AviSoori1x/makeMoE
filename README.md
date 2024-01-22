# makeMoE

<div align="center">
    <img src="images/makemoelogo.png" width="300"/>
</div>


From scratch implementation of a sparse mixture of experts language model inspired by (and largely based on) Andrej Karpathy's makemore :)

This is a from scratch implementation of a sparse mixture of experts language model. This is inspired by and largely based on Andrej Karpathy's project makemore and borrows most of the re-usable components from that implementation. Just like makemore, makeMoE is also an autoregressive character-level language model that uses the aforementioned sparse mixture of experts architecture. 

Just like makemore, pytorch is the only requirement (so I hope the from scratch claim is justified)

Significant Changes from the makemore architecture

- Sparse mixture of experts instead of the solitary feed forward neural net. 
- Top-k gating and noisy top-k gating implementations.
- initialization - Kaiming He initialization used here but the point of this notebook is to be hackable so you can swap in Xavier Glorot etc. and take it for a spin.

Unchanged from makemore
- dataset, preprocessing (tokenization), and the language modeling task Andrej chose originally - generate Shakespear-like text
- Casusal self attention implementation 
- Training loop
- inference code

Publications heavily referenced for this implementation: 
- Mixtral of experts: https://arxiv.org/pdf/2401.04088.pdf
- Outrageosly Large Neural Networks: The Sparsely-Gated Mixture-Of-Experts layer: https://arxiv.org/pdf/1701.06538.pdf

0_makMoE_intuition.ipynb walks through the intuition for the entire model architecture how everything comes together
1_makeMoE.ipynb is the consolidated hackable implementation that I encourage you to hack, understand, improve and make your own

Hope you find this useful and enjoy!!
