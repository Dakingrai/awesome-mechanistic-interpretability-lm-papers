# awesome-mechanistic-interpretability-LM-papers

This is a collection of awesome papers about Mechanistic Interpretability (MI) for Transformer-based Language Models (LMs), organized following our survey paper: [A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models](https://arxiv.org/pdf/2407.02646). 

Papers are organized following our **taxonomy (Figure 1)**. 
We have also curated a **Beginner's Roadmap (Figure 2)** with actionable items for interested people using MI for their purposes.

<div align="center">
  <img src="images/taxonomy.png" width="70%"/>
  <p>Figure 1: Taxonomy</p>
</div>

<div align="center">
  <img src="images/roadmap.png" width="50%"/>
  <p>Figure 2: Beginner's Roadmap</p>
</div>

**How to Contribute:** We welcome contributions from everyone! If you find any relevant papers that are not included in the list, please categorize them following our taxonomy and submit a request for update.


**Questions/Comments/Suggestions:** If you have any questions/comments/suggestions to share with us, you are welcome to report an issue here or reach out to us through drai2@gmu.edu and ziyuyao@gmu.edu.

**How to Cite:** If you find our survey useful for your research, please cite our paper:
```
@article{rai2024practical,
  title={A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models},
  author={Rai, Daking and Zhou, Yilun and Feng, Shi and Saparov, Abulhair and Yao, Ziyu},
  journal={arXiv preprint arXiv:2407.02646},
  year={2024}
}
```

## Updates
- June 2024: GitHub repository launched! Still under construction.

## Table of Contents
- [Techniques](#mi-techniques)
- [Evaluation](#evaluation)
- [Findings and Applications](#findings)
  - [Findings on Features](#features)
  - [Findings on circuits](#circuits)
    - [Interpreting LM Behaviors](#lm-behavior-interpret)
    - [Interpreting Transformer Components](#transformer-component-interpret)
  - [Findings on Universality](#universality)
  - [Findings on Model Capabilities](#model-capability-interpret)
  - [Findings on Learning Dynamics](#learning-dynamics)
  - [Applications of MI](#mi-application)
- [Tools](#tools)

## Paper Collection

### Techniques
|  Paper  |      Techniques    |      TL;DR    | 
| :----- | :--------------: | :----- | 
|  [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)  |      Logit lens    |   The paper proposed the "logit lens" technique, which can be used to project intermediate activations onto the vocabulary space for interpretation.      |
|  [Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://arxiv.org/pdf/2203.14680)  |      Logit lens    |   The paper showed that the "logit lens" can be used to project the second-layer of feed-forward parameter matrices to vocabulary space for interpretation.    | 
### Evaluation
|  Paper  |      Evaluation    |      TL;DR    | 
| :----- | :--------------: |  :----- |
|  [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/pdf/2211.00593)  |      Faithfulness, Completeness, Minimality    |   The paper proposed ablation-based techniques for the faithfulness, completeness, and minimality evaluation of the discovered circuit.     |

### Findings and Applications
#### Findings on Features
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [Softmax Linear Units](https://transformer-circuits.pub/2022/solu/index.html)  |   Visualization     |      N/A    |     The paper investigated the impact of changing the activation function in LMs from ReLU to the softmax linear unit on the polysemanticity of neurons.      |
#### Findings on circuits
##### Interpreting LM Behaviors
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [In-context learning and induction heads.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)  |   Zero-ablation, Visualization      |      Faithfulness    |      The paper demonstrates the importance of induction heads for in-context learning.    |
##### Interpreting Transformer Components
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [A mathematical framework for transformer circuits](https://transformer-circuits.pub/2021/framework/index.html)  |   Visualization    |      N/A    |      The paper shows that the "Residual stream (RS)" of LMs can be viewed as a one-way communication channel that transfers information from earlier to later layers. Furthermore, the paper also showed that each attention head in the "Multi-headed attention (MHA)" sublayer in a layer operates independently and can be interpreted independently   |
#### Findings on Universality
|  Paper  | Techniques |      Evaluation    |     TL;DR     |
| :----- | :--------------: | :--------------: | :--------------: 
|  [Successor Heads: Recurring, Interpretable Attention Heads In The Wild](https://arxiv.org/pdf/2312.09230)  |      Visualization, Logit lens    |      N/A    | The paper identifies an interpretable set of attention heads, termed "successor heads", which perform incrementation in LMs (e.g., Monday -> Tuesday, second -> third) across various scales and architectures. 
#### Findings on Model Capabilities
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [A mathematical framework for transformer circuits](https://transformer-circuits.pub/2021/framework/index.html)  |   Visualization      |      N/A    |      The paper discovered a circuit that implements the task of detecting and continuing repeated subsequences in the input (e.g., Mr D urs ley was thin and bold. Mr D -> urs)    |
#### Findings on Learning Dynamics
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [In-context learning and induction heads.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)  |   Zero-Knockout, Visualization      |      Faithfulness    |      The paper shows that transformer-based LMs undergo a "phase change" early in training, during which induction heads form and simultaneously in-context learning improves dramatically.    |
#### Applications of MI
|  Paper  |      Techniques    |      Evaluation    |      TL;DR    | 
| :----- | :--------------: | :--------------: | :--------------: | 
|  [Locating and Editing Factual Associations in GPT](https://arxiv.org/pdf/2202.05262)  |   Activation Patching      |      Faithfulness    |      The paper used activation patching to localize components that are responsible for storing factual knowledge, and then edited the fact (e.g., replacing "Seattle" with "Paris") by only updating the parameters of those components    |

### Tools
|  Paper  |      TL;DR    | 
| :----- | :--------------: |
|  [CircuitsVis](https://github.com/TransformerLensOrg/CircuitsVis)  |  Library for attention visualization      | 

