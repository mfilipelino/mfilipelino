# 📋 Practical Data Science on the AWS Cloud

#### Lectures notes [https://community.deeplearning.ai/t/pds-course-1-lecture-notes/48242](https://community.deeplearning.ai/t/pds-course-1-lecture-notes/48242)

<figure><img src=".gitbook/assets/Screen Shot 2023-10-29 at 2.42.49 PM.png" alt=""><figcaption><p>Machine Learning Workflow</p></figcaption></figure>

### Popular ML tasks and learning paradigmas

* **Supervised learning:** Classification & Regression&#x20;
* **Unsupervised** Clustering
* **Reinforcement learning**
* **Computer vision**
* **Text Analysis**

<figure><img src=".gitbook/assets/Screen Shot 2023-11-06 at 8.42.46 AM.png" alt=""><figcaption></figcaption></figure>

**The task:**

More specifically, you will perform multi-class classification for sentiment analysis of product reviews

<figure><img src=".gitbook/assets/Screen Shot 2023-10-26 at 2.45.35 PM (2).png" alt=""><figcaption></figcaption></figure>

## Data Bias and Feature Importance

* Describe the concept of data bias and compare popular bias metrics&#x20;
* Demonstrate how to detect data bias
* Understand feature importance

<figure><img src=".gitbook/assets/statisticalbias.png" alt="" width="188"><figcaption></figcaption></figure>

### Statistical bias

Statistical bias exists in numerous stages of the data collection and analysis process, including the source of the data, the methods used to collect the data, the [estimator](https://en.wikipedia.org/wiki/Estimator) chosen, and the methods used to analyze the data

Statistical bias occurs when a data set doesn't fully and accurately represent the situation it's supposed to reflect. It's like having a skewed perspective. When data has this kind of imbalance, it can lead to misleading conclusions or predictions.

Two examples are given in the text:



1. **Credit Card Fraud Detection:** If you're trying to build a model to detect credit card fraud, and your training data primarily consists of legitimate transactions, then your model might struggle to identify fraud. That's because it hasn't "seen" or "learned from" many fraudulent transactions. It's like trying to recognize a rare bird when you've only ever seen common ones. To fix this, you might need to add more examples of fraudulent transactions to your training data.
2. **Product Review Sentiment Analysis**: If you have a product review data set that mostly has reviews for one product category (let's call it "A") and very few for others ("B" and "C"), then a sentiment prediction model trained on this data will likely be good at predicting sentiments for products in category "A". However, it might perform poorly for products in categories "B" and "C". It's like being an expert in reviewing smartphones but not so knowledgeable about reviewing laptops or cameras.
3. **Healthcare Diagnosis Model:** Imagine you're developing a machine learning model to diagnose a particular disease, and your training data is sourced from hospitals in urban areas only. This data might not be representative of the broader population, particularly those in rural areas who might have different lifestyles, access to healthcare, and disease prevalence. It's akin to learning how to diagnose based on city dwellers and then trying to apply that knowledge to someone living in the countryside. To correct this, you'd need to include data from a variety of geographical locations.
4. **Job Applicant Screening Tool:** Suppose a company is using a machine learning model to screen job applicants. If the data used to train the model is mostly composed of successful applicants from a particular university or demographic, the model might develop a bias toward those candidates. It’s like having a preference for chocolate ice cream because that’s mostly what you’ve been offered, even though other flavors might be just as good. To mitigate this, the company would need to ensure a diverse range of successful applicants in their training data.
5.  **Ice Cream Flavor Recommendation System:**

    Imagine you're developing a recommendation system for an ice cream shop. The shop has historically sold mostly chocolate and vanilla flavours, so your dataset for training the recommendation system is heavily skewed toward these two flavors. Consequently, the system might start to recommend chocolate or vanilla more often to new customers, even if they might prefer other flavors like strawberry or mint chocolate chip. It's like assuming everyone prefers the same flavors because those are the ones you sell the most. This could lead to a self-fulfilling prophecy where the shop continues to sell mostly chocolate and vanilla, not because they are universally preferred, but because they are most frequently recommended. To counter this bias, you would need to either collect more balanced data reflecting all available flavours or adjust the recommendation algorithm to account for the underrepresentation of certain flavours.

In all cases, the bias in the data leads to models that might not work as expected, especially in scenarios that they aren't well-prepared for. This can have consequences for businesses, from making poor decisions based on the model's predictions to facing regulatory issues.

{% embed url="https://www.youtube.com/watch?v=PdXDLNNXPik" %}

### Statistical bias causes

1. **Activity Bias (Social Media Content)**: This arises from human-generated content, especially on social media. A small percentage of the population actively participates on these platforms, so the data collected is not representative of the entire population.
2. **Societal Bias (Human-generated content)**: This bias is present in data generated by humans, both on and off social media. It emerges from pre-existing societal notions. Since everyone has unconscious biases, the data they produce can reflect these biases.
3. **Feedback Loops (Selection bias)**: Bias can sometimes be introduced by the machine learning system itself. For instance, if a machine learning application provides users with options, and then uses their selections as training data, it can create feedback loops. An example given is a streaming service recommending movies based on a user's previous selections and ratings, which might not always capture the user's true preferences.
4. **Data Drift**: After a model is trained and deployed, the data it encounters in the real world can differ from the training data. This change in data distribution is known as data drift or data shift. There are different types:
   * **Covariant Drift**: When the distribution of the independent variables or features changes.
   * **Prior Probability Drift**: When the distribution of the labels or target variables changes.
   * **Concept Drift (or Concept Shift)**: When the relationship between features and labels changes. An example provided is the different terms used for soft drinks across various regions.
5. **Human Interpretation and Regional Differences**: The example of soft drinks being called "soda" in some areas and "pop" in others highlights how human interpretation and regional differences can introduce variations in labels and data.

Given these potential sources of bias, it's vital to monitor and detect biases in datasets continuously, both before and after training models. The focus is on identifying these biases and imbalances in the pre-training datasets.



### What are facets?&#x20;

A facet of a dataset refers to a specific feature or attribute within that dataset that is of particular interest, especially when analyzing for imbalances, biases, or other specific characteristics. In essence, a facet is a dimension or aspect of the data that one wants to examine more closely or treat as a sensitive attribute. Analyzing facets allows for a more granular understanding of how different attributes or features may be distributed or represented within the dataset.

Examples of facets

1. **Gender as a Facet**: If you have a dataset of employees, you might treat gender as a facet to examine for pay equity. Analyzing salaries across this facet could reveal gender pay gaps.
2. **Age Group as a Facet**: In a dataset of users for a social media platform, age group could be a facet. You might analyze engagement or content preferences across different age groups to understand usage patterns or to detect any age-related biases in content recommendation algorithms.
3. **Geographic Location as a Facet**: For a dataset containing medical records, geographic location could be a facet. Analyzing treatment outcomes across different regions could help identify if certain areas lack access to specific medical resources, indicating a geographical bias.
4. **Education Level as a Facet**: In a dataset of job applicants, education level might be a facet of interest. By examining the hiring rates across different education levels, you could assess if there's an unintentional bias towards candidates with a certain educational background.
5. **Ethnicity as a Facet**: If you're working with a dataset of loan applications, ethnicity might be a facet you'd want to examine closely. Analyzing loan approval rates across different ethnic groups can help uncover any potential biases in the loan approval process.

### Measuring Statistical Bias

Statistical bias and imbalances in datasets can arise due to various reasons. To quantify these imbalances, specific metrics target different facets of your dataset. A facet refers to a sensitive or important feature in your dataset you want to assess for imbalances. For instance, in a product review dataset, the product category might be a facet of interest.

Two key metrics are:

1. **Class Imbalance (CI)**: This metric gauges the disparity in the number of examples for different facet values. In the context of a product review dataset, CI would determine if a specific product category, like Category A, has a significantly larger number of reviews compared to other categories.

<figure><img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*zB0xRorLnqPzjMY_DQ3HAA.png" alt=""><figcaption></figcaption></figure>



1. **Difference in Proportions of Labels (DPL)**: DPL evaluates the imbalance in positive outcomes between various facet values. Using the product review example, DPL checks if a particular category, such as Category A, has notably higher ratings than others. In contrast to CI, which focuses on the quantity of reviews, DPL concentrates on the quality (ratings) of those reviews.



[Fairness Measures for Machine Learning](https://pages.awscloud.com/rs/112-TZM-766/images/Fairness.Measures.for.Machine.Learning.in.Finance.pdf)

[Measure pre-training bias - Amazon Sagemaker](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-measure-data-bias.html)

### Approaches to Statistical Bias Detection

There are two primary tools for detecting statistical bias in datasets: SageMaker Data Wrangler and SageMaker Clarify.

1. **SageMaker Data Wrangler**:
   * **Approach**: Provides a UI-based visual experience.
   * **Use Cases**: Suitable for users who prefer visually exploring data, connecting to multiple data sources, and configuring bias reports using dropdowns and option buttons. It allows the launching of bias detection jobs with a button click.
   * **Limitation**: Uses only a subset of your data for bias detection.
2. **SageMaker Clarify**:
   * **Approach**: Offers an API-based method.
   * **Features**: It can scale out the bias detection process using a construct known as processing jobs, which lets users configure a distributed cluster for executing bias detection at a larger scale.
   * **Use Cases**: Ideal for analyzing large data volumes, like millions of product reviews, to detect bias. It leverages the scalability and capacity of the Cloud.

### What is feature importance?&#x20;

In machine learning (ML) engineering, "feature importance" refers to a method or metric that helps determine the significance or contribution of individual features (or variables) to the predictive power of a model. Understanding feature importance is crucial for several reasons:

1. **Model Interpretability**: Knowing which features are most influential helps in understanding how the model makes its decisions, which is crucial for explaining the model's behaviour to stakeholders.
2. **Feature Selection**: If certain features have minimal or no importance, they might be excluded from the model, simplifying it and potentially improving its performance by reducing overfitting.
3. **Domain Knowledge Validation**: Feature importance can be used to validate if the model's decisions align with domain expertise. For instance, in a real estate price prediction model, one would expect features like "location" and "square footage" to be of high importance.
4. **Model Debugging**: If an irrelevant feature appears to be highly important, it might indicate issues with data quality or model training.
5. **Resource Optimization**: Collecting data for certain features can be expensive or time-consuming. If a feature is of low importance, resources can be reallocated more efficiently.

Various algorithms provide different methods to calculate feature importance. For example:

* **Tree-based models** (like Decision Trees, Random Forests, and gradient-boosted trees) have built-in methods to report feature importance based on how frequently a feature is used to split the data and its impact on model accuracy.
* **Linear models** can use the magnitude of coefficients as a measure of feature importance.
* **Permutation Importance**: It involves shuffling one feature's values and measuring the decrease in model performance. A significant decrease indicates high feature importance.
* **SHAP (Shapley Additive exPlanations)** values provide a unified measure of feature importance and fair allocation of contribution to each feature.

{% embed url="https://shap.readthedocs.io/en/latest/" %}

{% embed url="https://www.youtube.com/watch?v=cTa5HYCxTVg&t=7s" %}



## Automated Machine Learning - AutoML&#x20;

* Ability to reduce  Time to market&#x20;
* Lack of ML skillsets&#x20;
* Ability to iterate quickly&#x20;
* Ability to optimize scarce resources for more challenging use cases

SageMaker Autopilot will inspect the raw dataset, apply feature processors, pick the best set of algorithms, train and tune multiple models, and then rank the models based on performance - all with just a few clicks. Autopilot transparently generates a set of Python scripts and notebooks for a complete end-to-end pipeline including data analysis, candidate generation, feature engineering, and model training/tuning.

SageMaker Autopilot job consists of the following high-level steps:

* _Data analysis_ where the data is summarized and analyzed to determine which feature engineering techniques, hyper-parameters, and models to explore.
* _Feature engineering_ where the data is scrubbed, balanced, combined, and split into train and validation.
* _Model training and tuning_ where the top performing features, hyper-parameters, and models are selected and trained.

<figure><img src=".gitbook/assets/autoML.png" alt=""><figcaption><p>AutoML</p></figcaption></figure>

1. Extract

```
aws s3 cp 's3://dlai-practical-data-science/data/balanced/womens_clothing_ecommerce_reviews_balanced.csv' ./
```

```python
path_autopilot = './womens_clothing_ecommerce_reviews_balanced_for_autopilot.csv'

df[['sentiment', 'review_body']].to_csv(path_autopilot, 
                                        sep=',', 
                                        index=False)
autopilot_train_s3_uri = sess.upload_data(bucket=bucket, key_prefix='autopilot/data', path=path_autopilot)
autopilot_train_s3_uri
```

```python
aimport time

timestamp = int(time.time())
auto_ml_job_name = 'automl-dm-{}'.format(timestamp)
model_output_s3_uri = 's3://{}/autopilot'.format(bucket)

max_candidates = 3

automl = sagemaker.automl.automl.AutoML(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    target_attribute_name="sentiment", # Replace None
    base_job_name=auto_ml_job_name, # Replace None
    output_path=model_output_s3_uri, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    max_candidates=max_candidates,
    sagemaker_session=sess,
    role=role,
    max_runtime_per_training_job_in_seconds=1200,
    total_job_runtime_in_seconds=7200
)

automl.fit(
    ### BEGIN SOLUTION - DO NOT delete this comment for grading purposes
    autopilot_train_s3_uri, # Replace None
    ### END SOLUTION - DO NOT delete this comment for grading purposes
    job_name=auto_ml_job_name, 
    wait=False, 
    logs=False
)
```

Autopilot job status



[Autogluon](<README (1).md>)



{% embed url="https://www.amazon.science/publications/amazon-sagemaker-autopilot-a-white-box-automl-solution-at-scale" %}
paper
{% endembed %}

## Built-in algorithms

* Summarize why and when to choose built-in algorithms
* Describe the use case and algorithms
* Understand the evolution of text analysis algorithms
* Discuss word2vec, FastText and BlazingText algorithms
* Transform raw review data into features to train a text classifier
* Apply the Amazon SageMaker built-in BlazingText algorithm to train a text classifier
* Deploy the text classifier and make predictions

<figure><img src=".gitbook/assets/Screen Shot 2023-11-06 at 8.39.43 AM.png" alt=""><figcaption></figcaption></figure>

### Why use built-in algorithms?

* Implementation is highly highly-optimized and scalable (shift between CPU tp GPU as simple as a parameter of type of machine)&#x20;
* Focus more on domain-specific tasks rather than managing low-level model code and infrastructure
* Trained model can be downloaded and re-used elsewhere

<figure><img src=".gitbook/assets/Screen Shot 2023-11-06 at 8.40.36 AM.png" alt=""><figcaption></figcaption></figure>

### ML types tasks and built-in algorithms

<figure><img src=".gitbook/assets/Screen Shot 2023-11-06 at 8.42.46 AM.png" alt=""><figcaption></figcaption></figure>



### Classification and Regression - Tabular data



<table><thead><tr><th width="318">Example problems and use cases</th><th width="237.33333333333331">Problem types</th><th>Built-in algorithms</th></tr></thead><tbody><tr><td>Predict if an item belongs to a category: an email spam filter</td><td>Binary/multi-class classification</td><td>XGBoost, K-Nearest Neighbors</td></tr><tr><td>Predict a numeric/continuous value: estimate the value of a house</td><td>Regression</td><td>Linear Learner, XGboost</td></tr><tr><td>Predict sales on a new product based on previous sales data</td><td>Time-series forecasting</td><td>DeepAR Forecasting</td></tr></tbody></table>

### Clustering

| Example problem and use cases                                             | Problema type                          | Built-in algorithms                                         |
| ------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------- |
| Drop weak features such as the color of a car when predicting its mileage | Feature enginnering: reduce dimensions | Principal component analysis (PCA)                          |
| Detect abnormal behavior                                                  | Anomaly detection                      | Random cut forest (RCF)                                     |
| Group high/medium/low-spending customer from transaction histories        | Clustering / Grouping                  | K-Means                                                     |
| Organize a set of documents into topics based on words and phares         | Topic modeling (NLP)                   | Latent Dirichlet Allocation (LDA), Neural Topic Model (NTM) |
|                                                                           |                                        |                                                             |
|                                                                           |                                        |                                                             |

### Image processing

| Example problems and uses cases                 | Problem types        | Buil-in algoritms                                       |
| ----------------------------------------------- | -------------------- | ------------------------------------------------------- |
| Content moderation                              | Image classification | Image classification (full training, transfer learning) |
| Detect people and objects in an image           | Object detection     | Object detection                                        |
| Self-driven cars identify objects in their path | Computer vision      | Semantic Segmentation                                   |

### &#x20;Text analysis

| Example problems and use cases       | Problem types       | Built-in algoritms   |
| ------------------------------------ | ------------------- | -------------------- |
| Convert spanish to english           | machine translation | sequence-to-sequence |
| Summarize a research paper           | Text sumarization   | sequence-to-sequence |
| Transcribe call center conversations | Speech-to-text      | Sequence-to-sequence |
| Classify reviews into categories     | Text classification | Blazing Text         |

### Evolution of text analysis algorithms&#x20;

<figure><img src=".gitbook/assets/Screen Shot 2023-11-09 at 9.22.11 AM.png" alt=""><figcaption></figcaption></figure>

### Additional reading material

* [Word2Vec algorithm](https://arxiv.org/pdf/1301.3781.pdf)
* [GloVe algorithm ](https://www.aclweb.org/anthology/D14-1162.pdf)
* [FastText algorithm ](https://arxiv.org/pdf/1607.04606v2.pdf)
* [Transformer architecture, "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
* [BlazingText algorithm](https://dl.acm.org/doi/pdf/10.1145/3146347.3146354)
* [ELMo algorithm ](https://arxiv.org/pdf/1802.05365v2.pdf)
* [GPT model architecture ](https://cdn.openai.com/research-covers/language-unsupervised/language\_understanding\_paper.pdf)
* [BERT model architecture](https://arxiv.org/abs/1810.04805)
* [Built-in algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
* [Amazon SageMaker BlazingText](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html)

## Build, Train, and Deploy ML Pipelines using BERT



### Feature Engineering and Feature Store

#### Learning Objectives

* Describe the concept of feature engineering
* Apply feature engineering to prepare datasets for training
* Understand the importance of a feature store
* Demonstrate how to store and share features using Amazon SageMaker Feature Store



