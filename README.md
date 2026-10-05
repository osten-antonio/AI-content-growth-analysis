# Estimating the Percentage Growth of AI Content Across Categories of the Web 
> Published at ICISS 2026 · [DOI: 10.1109/ICISS71466.2026.11715500](https://doi.org/10.1109/ICISS71466.2026.11715500)

## Overview
Generative AI tools have made it increasingly hard to tell whether web content such as blog posts, news articles, social media posts, was written by a person or a machine. This matters as AI systems are known to hallucinate, and as more of the content people consume is AI-generated, the risk of misinformation spreading at scale grows too [1], [2], which makes it important to understand how much AI content exists and how fast it's growing, especially in domains like news which shapes people's understanding on the ongoing real world things.
 
Most existing research either focuses narrowly on one content type, such as news [3], or gives a single aggregate estimate [4]. However, none of it tracks how the share of AI-generated content is changing over time.
 
This project tackles that gap with a two-phase pipeline:
 
- **Classification** - training and comparing traditional ML (SVC, Random Forest) against deep learning (BERT, LSTM) models to label text as AI-generated or human-written, using stylometric and text-based features.
- **Forecasting** - using the labeled data to estimate monthly AI-content percentages per category from 2020-2025, then comparing linear and non-linear regression models to project growth trends.
The result is a category-level view of how much AI-generated content shows up in blogs, news, and social media, how fast that share is growing, and what that means for information integrity going forward.
 
 
## Research questions
- To what extent does AI-generated content appear across different platforms, sources, and domains on the web (news, social media, blogs)?
- How much is the growth of these AI generated content across these domains?
- What type of models are more suitable for classifying AI vs Human text?
- What type of regression models are good when fitting real time noisy amounts of AI content data?
## Data
 
- **Classifier training**: [AI Vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) (Kaggle)
- **Cross-domain validation**: [Human vs. LLM Text Corpus](https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus/code) (Kaggle)
- **Blogs**: Gizmodo, Lifehacker, and Machine Learning Mastery articles (scraped), plus [Medium-Articles-Corpus](https://huggingface.co/datasets/crawlfeeds/Medium-Articles-Corpus) (HuggingFace)
- **News**: [financial-news-multisource](https://huggingface.co/datasets/Brianferrell787/financial-news-multisource) (HuggingFace)
- **Social media**: [YouTube-Commons](https://huggingface.co/datasets/Rijgersberg/YouTube-Commons), [reddit_dataset_94](https://huggingface.co/datasets/coldmind/reddit_dataset_94), [twitter100m_tweets](https://huggingface.co/datasets/enryu43/twitter100m_tweets), [twitter-text-dataset](https://huggingface.co/datasets/bittensor-dataset/twitter-text-dataset), and a sample of Facebook posts ([source](https://github.com/luminati-io/Facebook-dataset-samples))

All sources cover 2020-2025.
 
## Results
 
- **AI-content share**: news and social media averaged ~60%, blogs averaged 20-30%
- **Monthly growth (2020-2025)**: blogs +0.14% (p<0.001), news +0.078% (p=0.007), social media +0.032% (not significant, p=0.22)
- **Best classifier**: Random Forest (traditional ML) outperformed BERT/LSTM on cross-domain generalization
- **Best regression model**: ElasticNet outperformed non-linear alternatives for trend forecasting
## References
[1] R. Zellers, A. Holtzman, H. Rashkin, Y. Bisk, A. Farhadi, F. Roesner,
Y. Choi, and P. Allen, "Defending against neural fake news," 12 2020.
[Online]. Available: https://arxiv.org/pdf/1905.12616
 
[2] S. Park and X. Nan, "Generative AI and misinformation: a scoping
review of the role of generative AI in the generation, detection, mitigation,
and impact of misinformation," *AI & Society*, 09 2025.
 
[3] H. W. A. Hanley and Z. Durumeric, "Machine-made media: Monitoring
the mobilization of machine-generated articles on misinformation
and mainstream news websites" arXiv.org, 2023. [Online]. Available:
https://arxiv.org/abs/2305.09820
 
[4] D. H. Spennemann, "Delving into: the quantification of AI-generated
content on the internet (synthetic data)" arXiv.org, 2025. [Online].
Available: https://arxiv.org/abs/2504.08755
