# Language-Based Classifier for OOV Generalization

## Abstract

Large Language Models (LLMs) have shown remarkable performance in various Natural Language Processing (NLP) tasks. Recently, their application has extended to analyzing tabular data. Despite their success in NLP tasks like text classification, their adoption for tabular data classification has been relatively limited, primarily due to their inferior performance compared to traditional machine learning models (TMLs), such as XGBoost. However, LLMs possess a unique potential for tabular data classification tasks by leveraging the context between variables based on pre-trained knowledge. This capability suggests that LLMs can interpret data contexts that are typically challenging to learn due to a significant number of missing values in tabular data or the presence of new variables not seen during training, referred to as out-of-variable (OOV) tasks.

We propose a methodology named Language-Based Classifier (LBC), which uniquely addresses the OOV tasks in tabular data classification, distinguishing itself from TMLs. LBC capitalizes on its ability to handle OOV tasks through a novel approach to tabular data classification. This approach involves converting tabular data into natural language prompts, allowing LBC to seamlessly and intuitively manage OOVs for inference. Moreover, the interpretation of OOVs using LBC's pre-trained knowledge base aids in increasing the likelihood of correctly classifying the answer class. Employing three key methodological strategies — Categorical Changes to adjust data for better model comprehension, Advanced Order and Indicator to enhance data representation, and the use of a Verbalizer to map logit scores to classes during inference — LBC emphasizes its capability to effectively tackle OOV tasks. We empirically and theoretically demonstrate the superiority of LBC, marking it as the first study to apply an LLM-based model to OOV tasks.

![LBC Methodology Overview](path/to/your/image.jpg)

*Figure 1: Overview of the Language-Based Classifier (LBC) methodology.*

size the model’s ability to effectively handle OOV tasks. We empirically and theoretically validate the superiority of LBC. LBC is the first study to apply an LLM-based
model to OOV tasks.
