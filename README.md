# CS677-Machine-Learning
Repository for storing code for my Pace University course CS677 Machine Learning.

## Discussion Posts

Here are some of the discussion posts I submitted in response to various prompts.
- [ML Applications](#ml-applications)
- [Data Quality for ML](#data-quality-for-ml)
- [Artificial General Intelligence](#artificial-general-intelligence)
- [ML Opinions](#ml-opinions)

### ML Applications
**Discuss the various ML applications and their impact on society.**

Machine learning is an ever-expanding field of research and applications, and has improved every situation where it has been applied. In today's fast-paced world, it is estimated that data on the order of zettabytes (~100 billion gigabytes) is being created every year. This is a lot of data which can provide a lot of insight into our society, our world, and even the universe. Machine learning, artificial intelligence, deep learning, and many other techniques are being applied across a range of disciplines to analyze these data, and has been yielding fruitful results.

For example, applications of ML such as computer vision and pattern recognition in the medical sector have been revolutionary for predicting diseases, tumors, etc. from medical images, allowing for more targeted healthcare for patients, and saving lives.

ML applications like forecasting, regression, classification, etc. are also used very heavily in the finance/economic sectors for credit card fraud detection, banking risk assessments, stock trading, global currency trading, etc. This helps those institutions make informed decisions on consumer lending, trading, etc.

Having done my undergraduate degree in astrophysics, one example of ML in that field is computer vision. There are networks of telescopes all around the world (and in space) taking terabytes of images each night. Computer vision and pattern detection have been very instrumental in auto-detecting very small and faint objects in these images, like comets and asteroids, in real time. This enables astrophysicists to focus on doing scientific research on these comets and asteroids without having to spend a lot of time in astrophysical image processing (which can be very time-consuming). This benefits our society by allowing us to understand our origins better in the context of planetary formation. 

This is just a very tiny snapshot of how machine learning applications have beneficial impacts in our society across many fields. 

### Data Quality for ML
**Read the document attached and post your view in 2 paragraphs.**
**[Overview and Importance of Data Quality for Machine Learning.pdf](https://research.ibm.com/publications/overview-and-importance-of-data-quality-for-machine-learning-tasks)**

This paper is a tutorial abstract written by researchers from IBM India for the 'Knowledge Discovery and Data Mining' conference in 2020. In the paper, they describe several approaches that can be taken during the machine learning process to ensure good data quality, and how to measure the data quality. Some of their proposed data quality metrics include label noise (how to reduce noise caused by erroneous labels), class imbalance (how to address imbalanced data), data valuation (how to use data valuation), data homogeneity (how to detect heterogeneity in data), data transformation (how to use better data transformations), and data cleaning (techniques on how to clean data better). They also talk about data quality metrics for unstructured data, for example generic text quality approaches (how to learn different types of text quality), optimizing quality metrics (with respect to the textual classification task and having labels), quality metrics for corpus filtering (deriving text quality of samples and using them for filtering), and outlier detection. Finally, they also mention the importance of having a human Subject Matter Expert (SME) in being involved in the data quality assurance process. 

I think this is a great paper for us to learn about data quality issues and how to deal with them, so that we can put the best data possible into our models and receive good results back. I always think about the 'garbage in, garbage out' paradigm when dealing with data, because if we are putting bad data (e.g. missing/noisy labels, class imbalances, bad transformations and cleaning, etc.) into our machine learning model, it will train the predictor to learn from those bad samples, and output inaccurate results, amongst other issues. The same applies for unstructured data as well. I definitely agree with the authors that it is very important to have a human SME verifying the data quality based on the metrics results in order to make the data pipeline more reliable. I think that as beginners in this field, it's very easy to get excited by the hype of machine learning, and quickly putting data in to obtain results. However, as data scientists, it is one of our moral and ethical duties to make sure that the data we are using at all stages of the machine learning process - start to end and beyond - is actually fit for consumption in a manner that can be measured and compared over time, such as through the techniques proposed in this paper.

### Artificial General Intelligence
**Read the complete information and write a summary of your understanding and viewpoint in not less than 150 words. https://en.wikipedia.org/wiki/Technological_singularity**

Artificial general intelligence is the idea of a superhuman intelligence being developed which has the capability of surpassing human intelligence and leading to a technological singularity, at which point technological growth will accelerate and lead to unforeseen changes to the human civilization. It could potentially improve itself autonomously over iterations, for example its own hardware and software, such that eventually its computing power approaches infinity on a human timescale (hence the 'singularity').

Currently I don’t think we are anywhere near this capability, although the general state of AI is definitely improving by leaps and bounds, especially with the advent of tools like ChatGPT. I think we might be close to reaching this point of a technological singularity when an AI is able to undoubtedly pass the Turing Test, for one. The Turing Test was proposed by Alan Turing in 1950, and essentially states that if a human interrogator is asking some written questions to a computer, and cannot tell if it's a human replying back or the computer, then the Turing Test will be passed (I learned about this in CS627 Artificial Intelligence class I took last semester!). 

However, the Turing Test says nothing about the actual intelligence of the machine, as in: is the computer actually aware of what it is doing, or has it simply learned the rules and derived a simple model of how the world works from its perspective, based on the data (machine learning)? From what I've seen, technology like neural networks aim to replicate (to some extent) the functionality of the human brain, which is sentient and aware of itself ("I think, therefore I am"), however there is no evidence that the computer is able to think sentiently. Once this happens, that will signal the start of the technological singularity, because as they point out in the article, the pace of sentient AI will accelerate and eventually reach a point where it surpasses human intelligence. 

This concept reminds me of this story by Isaac Asimov (very thought-provoking!): https://astronomy.org/moravian/C00-Last%20Question.pdf

### ML Opinions
**Discuss your opinion about each of the topics below in 200 words. Automated ML, Embedded ML, Centralized ML, Reproducible ML, Federated ML.**