# 📋 Practical Data Science on the AWS Cloud

#### Lectures notes [https://community.deeplearning.ai/t/pds-course-1-lecture-notes/48242](https://community.deeplearning.ai/t/pds-course-1-lecture-notes/48242)

<figure><img src=".gitbook/assets/Screen Shot 2023-10-29 at 2.42.49 PM.png" alt=""><figcaption><p>Machine Learning Workflow</p></figcaption></figure>

#### Common ML tasks

* **Supervised learning:** Classification & Regression&#x20;
* **Unsupervised** Clustering
* **Reinforcement learning**



**The task:**

More specifically, you will perform multi-class classification for sentiment analysis of product reviews

<figure><img src=".gitbook/assets/Screen Shot 2023-10-26 at 2.45.35 PM (2).png" alt=""><figcaption></figcaption></figure>

### Data Bias and Feature Importance

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
