# Final year Project

## Toxic Comment Detection (on Twitter) | Python, Flask, Tweepy, NLP Libraries 

################################################################

- Preprint on EasyChair: https://www.easychair.org/publications/preprint/lVNn

- Code Implementation repo: https://github.com/jgdshkovi/final_yr_proj

<!-- Not Deploement Ready - Live preview is not available for this project, as it requires the GPU to be running in the background -->
################################################################

## TLDR - Summary
- Successfully fine-tuned the BERT model (transfer learning) by leveraging the Jigsaw Toxic comment dataset from
Kaggle, enhancing its performance for accurate toxic comment classification.
- Implemented an integrated system that extracts tweets from the user’s timeline through the Twitter API and
seamlessly passes them to the trained NLP model for analysis.
- Classified tweets using the trained model, evaluating their toxicity scores, and further taking appropriate actions
based on the returned scores, contributing to a safer online environment.

################################################################

## Too Long Do Read

Toxic comments are disrespectful, abusive. Unreasonable online comments that usually make other users leave a discussion. The danger of online bullying and harassment affects the free flow of thoughts by restricting the dissenting opinions of
people. Sites struggle to promote discussions effectively, leading many communities to limit or close down user comments altogether. 

We will systematically examine the extent of online harassment and classify the content into labels to examine the toxicity as correctly as possible. We will aim at examining the toxicity with high accuracy to limit down its adverse effects which will be an incentive for organizations to take the necessary steps like reporting the user or blocking the user.

Online hate, described as abusive language, insults, personal attacks, threats or toxicity, has been identified as a major threat on online social media platforms. There are many billions of text data that’s being generated every day by in-apps messages, social media platforms, forums, blogs etc. All these channels are constantly generating large amounts of text data every second. Because of the large volumes of text data as well as unstructured data sources, we can no longer use the common approach to understand the text and this is where NLP comes in. With the increasing amount of text data being generated every day, NLP will only become more and more important to make sense of the data and used in many applications. 

Social Media Platforms (SMPs) are the most prominent grounds of toxic behaviour. Even though they provide ways to flag offensive and toxic content, only 17% of all the adults have flagged harassment conversations, and only 12% have reported someone of such acts. 

Manual techniques like flagging are neither effective nor easily scalable and have a risk of discrimination under subjective judgments by human annotators. Since an automated system can be much faster than human footnotes, machine learning and deep learning models to automatically detect online hate have been gaining popularity and bringing researchers from different fields together. 

To address these concerns, we propose to develop an online hate classifier using state-of-the-art NLP models like Bidirectional Encoder Representations from Transformers (BERT), GPT (Generative Pre-trained Transformer) etc,. We perform transfer learning, making use of pretrained models, which reduces the cost and time for training.


