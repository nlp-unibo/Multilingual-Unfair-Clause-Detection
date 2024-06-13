# Multilingual-Unfair-Clause-Detection

Repository for the paper [**Unfair clause detection in terms of service across multiple languages** (Galassi et al. 2024)](https://doi.org/10.1007/s10506-024-09398-7)

## Abstract
Most of the existing natural language processing systems for legal texts are developed for the English language. Nevertheless, there are several application domains where multiple versions of the same documents are provided in different languages, especially inside the European Union. One notable example is given by Terms of Service (ToS). In this paper, we compare different approaches to the task of detecting potential unfair clauses in ToS across multiple languages. In particular, after developing an annotated corpus and a machine learning classifier for English, we consider and compare several strategies to extend the system to other languages: building a novel corpus and training a novel machine learning system for each language, from scratch; projecting annotations across documents in different languages, to avoid the creation of novel corpora; translating training documents while keeping the original annotations; translating queries at prediction time and relying on the English system only. An extended experimental evaluation conducted on a large, original dataset indicates that the time-consuming task of re-building a novel annotated corpus for each language can often be avoided with no significant degradation in terms of performance.

## Cite

Please cite our work as
*Andrea Galassi, Francesca Lagioia, Agnieszka Jabłonowska, Marco Lippi, "Unfair clause detection in terms of service across multiple languages", 2024*

```
@article{galassi2024unfair,
  title={Unfair clause detection in terms of service across multiple languages},
  author={Galassi, Andrea and Lagioia, Francesca and Jab{\l}onowska, Agnieszka and Lippi, Marco},
  journal={Artificial Intelligence and Law},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10506-024-09398-7},
  url={https://doi.org/10.1007/s10506-024-09398-7}
}
```
