# Deception Detection using Machine learning

We aim to develop a machine learning model capable of distinguishing between true and deceptive audio stories. This **classification** task falls under **supervised learning** where the goal is to predict binary labels, truth or lie, by using the audio features extracted from the given recordings.
Deception detection is an interesting problem to solve and has it's applications in law enforcement, behavioral study and security. Experts in the field can identify whether a person is lying by observing speech patterns, tone, etc. However, this is a manual process subject to human bias. Using machine learning, we can process large amounts of data in a matter of minutes and eliminate bias.
My classmates and I have collected a dataset of around 600 audio recordings, 300 true and 300 deceptive. Each of these recordings are around 2-4mins long. Out of these, we have selected 100 recordings (50 true and 50 deceptive) for this project. We have split these audio recordings into chunks of 30 seconds and then extracted audio features from these to train our machine learning models.

We first split all the audio files into chunks of **30 seconds**. If the last chunk of the audio file is not exactly 30 seconds, we checked it's length. If it's more than 10 seconds, we **overlapped** it with the previous chunk, otherwise, we **discarded** the chunk. We have implemented this using the split_audio function.

Once we got the 30 second chunks, we **reduced the noise** in all these files so that it is easier to extract important features. We used the noisereduce library for this task. Check the reduce_noise function for this implementation.
<br>Next, we have extracted below audio features using python libraries.
1. **Power:** The rate at which sound energy is emitted could be an indicator of decpetion. Hence, we have extracted this feature using the audio time series.
2. **Pitch:** High pitch could be another indicator of deception. If the mean of the pitch of the speaker is higher than the baseline, we can say that the speaker is lying. Also, if the variablity in the pitch is high, this could also suggest nervousness which can be related to deception. Hence we're calculating mean and standard deviation of the pitch.
3. **Voiced Fraction:** It is the likelihood that a specific audio frame contains voiced speech. This can help us calculate the fractions of audio that are voiced.
4. **Silence Ratio:** This is different from voiced fraction. This tells us the amount of pauses the speaker takes while speaking. This could be an indicator of deception as one takes more pauses (shows hesitations in speech) while lying. 
5. **Zero-Crossing Rate:** ZCR measures the rate at which a signal changes sign (crosses the zero amplitude level). Hesitations in speech or sudden bursts of energy while talking can cause variations in ZCR. These may correlate with stress associated while speaking lies. 
6. **MFCC Features:** Mel frequency cepstral coefficients represent the spectral characteristics of an audio signal. Cepstral coefficients are accurate in pattern recognition problems related to human voice. Most of the important information is available in the first few coefficients. Hence, we've chosen 13 features for this project based on references.

Once we got this dataset of features, we have split it into three parts: **train, validation and test**. We have left the test dataset aside, this is used directly at the end to test the final ML model.
We first split the entire dataset into 80:20 ratio as train and test. Then, we again split the training data (80% of total) into 85:15 ratio as train and validation.

**Independence:** Note that we're using **GroupShuffleSplit** to split our dataset. We gave a **group ID** to each audio file and made sure that chunks of one file do not get split into different datasets. This ensures independence as all the chunks of one audio file will be present in only one dataset.

**Scaling:** We're using **Standard Scaler** to scale all the features. This is because it transforms the mean to 0 and variance to 1. Many machine learning algorithms (e.g., logistic regression, SVMs, and neural networks) assume that the features are centered around zero and scaled to a similar range. StandardScaler ensures that all features contribute equally to the model by normalizing their magnitude. Without scaling, features with larger magnitudes might dominate the training process. We can use MinMaxScaler if we're going to use neural networks.

**Dimensionality Reduction:** Handling high dimensional data can be computationally challenging, hence we have performed dimensionality reduction using **PCA**. This will reduce redundant or less relevant features while still capturing the essential structure. This will also help to avoid overfitting to our train data.

*Overfitting: The model would perform very well on training data, but will show poor performance on the validation/test data.

Once we have the final data ready to be used, we can start training various models using the training data and test them on the validation data. Depending on the results, we can choose the best family of models. We will also change the parameters under the selected family of models and choose the one that gives us best results. 

Once we're happy with our model, we can then train it on the entire dataset (training plus validation) and test it on the **unseen** test data.

To choose the best performing model, we will plot the **Confusion Matrix** and print the **Classification Report**. These will help us get the below parameters:
<br>**Accuracy**: We will most likely go for a model with a high accuracy on the validation dataset, as long as it is not overfitting*. 
<br>**Precision**: Precision will show us how often our model correctly predicts the positive class. It is the number of True Positives by the total number of positives (True and False Positives).
<br>**Recall**: Recall will show us how many of the total positive instances were correctly predicted by the model.
<br>**F1-score:** It is the harmonic mean of precision and recall and is especially used in case of imbalanced datasets.

We tried using models such as Logistic Regression, SVM, Decision Tree Classifier and Random Forest Classifier. However, the simplest model of them all, **Logistic Regression** performed the best based on the calculated parameters. Hence, we have used Logistic Regression in this case to visualise our results. Additionally, as we do not have a properly balanced dataset in our case, we have tried to use the **'class_weight=balanced'** parameter from Logistic Regression. This gave us a slightly better result than without balancing. Note that **LinearSVC** also performed quite well for our problem, but Logistic Regression gave a slightly higher precision and accuracy.

<br> We have also tried different **solvers** under Logistic Regression like 'liblinear', 'saga', etc along with different **Regularization** techniques such as 'l1', 'l2' and 'elasticnet'. However, we saw similar results on these and opted to go for the default parameters. 

Note that we have also tried a few ensemble models like **Random Forest Classifier** and boosting techniques like **Adaboost Classifier** and **Gradient Boost Classifier**. However, these models seem too complex for the dataset at hand. As we have a small dataset, these models are overfitting on the training data and performing poorly when presented with validation dataset. Hence, I've opted not to go for any ensemble methods in the final code. 

**The results and conclusions are provided at the end of the code.**


**References:**

A. R. Bhamare, S. Katharguppe and J. Silviya Nancy, "Deep Neural Networks for Lie Detection with Attention on Bio-signals," 2020 7th International Conference on Soft Computing & Machine Intelligence (ISCMI), Stockholm, Sweden, 2020: https://ieeexplore.ieee.org/document/9311575

Xiaohe Fan, Heming Zhao, Xueqin Chen, Cheng Fan and Shuxi Chen, "Deceptive Speech Detection based on sparse representation," 2016 IEEE 12th International Colloquium on Signal Processing & Its Applications (CSPA), Malacca City, 2016: https://ieeexplore.ieee.org/document/7515793

Enhancing Lie Detection Accuracy: A Comparative Study of Classic ML, CNN, and GCN Models using Audio-Visual Features: https://arxiv.org/html/2411.08885v1

Shanjita Akter Prome, Neethiahnanthan Ari Ragavan, Md Rafiqul Islam, David Asirvatham, Anasuya Jegathevi Jegathesan, 
Deception detection using machine learning (ML) and deep learning (DL) techniques: A systematic review 
Natural Language Processing Journa ,
Volume 6,
: https://www.sciencedirect.com/science/article/pii/S2949719124000050

https://www.cs.columbia.edu/speech/cxd/deceptionpapers.html

https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce5324,
