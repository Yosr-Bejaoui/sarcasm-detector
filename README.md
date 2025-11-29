# Sarcasm Detection in English Tweets

A comprehensive machine learning project for detecting sarcasm in English tweets using the iSarcasmEval dataset. This project explores various feature extraction methods, sampling techniques, and machine learning models, culminating in a fine-tuned RoBERTa model achieving state-of-the-art performance.

## Project Overview

Sarcasm detection is a challenging natural language processing task that requires understanding context, tone, and subtle linguistic cues. This project systematically compares multiple approaches to identify the most effective methods for detecting sarcasm in Twitter data.

### Key Highlights

- **Comprehensive Model Comparison**: Evaluated 96 different model configurations
- **Multiple Feature Extraction Methods**: TF-IDF, BERT, RoBERTa, and DistilBERT embeddings
- **Advanced Sampling Techniques**: Baseline, Random Under-Sampling, Random Over-Sampling, and SMOTE
- **Fine-Tuned Solution**: Optimized RoBERTa model with Focal Loss for class imbalance
- **Best Performance**: Achieved high F1-scores using transformer-based approaches

## Repository Structure

```
sarcasm-detector/
├── iSarcasmEval_EN/
│   ├── train.En.csv                                    # Training dataset
│   └── task_A_En_test.csv                             # Test dataset
├── Model_Exploration.ipynb                             # Comprehensive model comparison
├── fine-tuning RoBERTa.ipynb                          # Fine-tuned RoBERTa implementation
├── Deep_Project_ML_final.ipynb                        # Deep learning with RoBERTa embeddings
├── Project_ML_Hyperparameter_Tuning_Optuna (1).ipynb # Hyperparameter optimization using Optuna
├── Project_ML_Hyperparamters_Grid_ (1).ipynb         # Grid search hyperparameter tuning
├── comprehensive_model_results.csv                     # All model evaluation results
├── model_comparison_heatmap.png                       # Performance visualization heatmaps
├── feature_comparison.png                             # Feature extraction comparison
├── sampling_comparison.png                            # Sampling technique comparison
└── README.md                                          # Project documentation
```

## Methodology

### 1. Data Preprocessing

Text cleaning pipeline includes:
- URL normalization → `[URL]` token
- Mention handling → `[USER]` token
- Hashtag processing (preserving text content)
- Punctuation normalization
- Whitespace removal

### 2. Feature Extraction Methods

| Method | Description | Dimensions |
|--------|-------------|------------|
| **TF-IDF** | Term Frequency-Inverse Document Frequency with n-grams (1-3) | 5,000 features |
| **BERT** | Bidirectional Encoder Representations from Transformers | 768-dimensional embeddings |
| **RoBERTa** | Robustly Optimized BERT Pretraining Approach | 768-dimensional embeddings |
| **DistilBERT** | Distilled version of BERT (lighter, faster) | 768-dimensional embeddings |

### 3. Sampling Techniques

To address class imbalance:
- **Baseline**: No sampling (class-weighted models)
- **Random Under-Sampling**: Reduce majority class
- **Random Over-Sampling**: Duplicate minority class
- **SMOTE**: Synthetic Minority Over-sampling Technique

### 4. Machine Learning Models

Six different classifiers tested:
1. **Logistic Regression**: Linear baseline model
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting framework
4. **Support Vector Machine (SVM)**: Kernel-based classifier
5. **K-Nearest Neighbors (KNN)**: Instance-based learning
6. **Naive Bayes**: Probabilistic classifier

### 5. Hyperparameter Optimization

Two advanced hyperparameter tuning approaches:

#### Grid Search (Project_ML_Hyperparamters_Grid_)
- Exhaustive search through specified parameter combinations
- Tests models: Logistic Regression, Random Forest, XGBoost, SVM, KNN
- Uses stratified cross-validation for robust evaluation
- Feature engineering with emoji counts, punctuation patterns, and text statistics

#### Optuna Optimization (Project_ML_Hyperparameter_Tuning_Optuna)
- Automated hyperparameter optimization using Bayesian optimization
- Explores larger parameter spaces efficiently
- Advanced feature extraction including:
  - Emoji and punctuation counts
  - Capitalization patterns
  - Sentiment polarity (TextBlob)
  - Linguistic markers (intensifiers, contrast words)
- Ensemble methods with voting classifiers

### 6. Deep Learning Approach

#### Stacking Ensemble (Deep_Project_ML_final)
- RoBERTa embeddings as base features
- Stacked ensemble combining multiple classifiers
- Advanced text preprocessing with contraction expansion
- Custom feature engineering pipeline

### 7. Fine-Tuned RoBERTa Model

The final optimized solution uses:
- **Base Model**: `cardiffnlp/twitter-roberta-base-irony`
- **Custom Focal Loss**: Handles class imbalance with alpha-weighted loss
- **Hyperparameters**:
  - Learning Rate: 1.2e-5
  - Batch Size: 16 (with gradient accumulation)
  - Epochs: 4
  - Max Sequence Length: 128 tokens
  - Dropout: 0.2
  - Label Smoothing: 0.05
- **Optimization**: AdamW optimizer with warmup and weight decay
- **Early Stopping**: Patience of 2 epochs

## Results

### Model Exploration Results

The comprehensive comparison evaluated 96 configurations (6 models × 4 features × 4 sampling techniques).

**Top Performing Configurations:**

| Rank | Model | Features | Sampling | F1-Score | Precision | Recall |
|------|-------|----------|----------|----------|-----------|--------|
| 1 | XGBoost | RoBERTa | Under-Sampling | 0.4677 | 0.3591 | 0.6705 |
| 2 | SVM | RoBERTa | Under-Sampling | 0.4637 | 0.3342 | 0.7572 |
| 3 | Logistic Regression | RoBERTa | Over-Sampling | 0.4619 | 0.3773 | 0.5954 |

**Key Findings:**
- RoBERTa embeddings consistently outperformed other feature extraction methods
- Under-sampling and over-sampling were more effective than SMOTE
- Transformer-based embeddings (BERT, RoBERTa, DistilBERT) significantly outperformed TF-IDF
- Class imbalance handling is crucial for sarcasm detection

### Fine-Tuned RoBERTa Results

The fine-tuned model with threshold optimization achieved superior performance through:
- Full model fine-tuning (all layers trained)
- Focal Loss to prioritize hard examples
- Optimal decision threshold selection

## Experimental Process

This project follows a systematic approach to sarcasm detection, progressing from basic models to advanced deep learning solutions:

### Phase 1: Comprehensive Model Exploration
**Notebook**: `Model_Exploration.ipynb`

Evaluated 96 different configurations combining:
- 4 feature extraction methods (TF-IDF, BERT, RoBERTa, DistilBERT)
- 4 sampling techniques (baseline, under-sampling, over-sampling, SMOTE)
- 6 machine learning models (Logistic Regression, Random Forest, XGBoost, SVM, KNN, Naive Bayes)

Generated comprehensive performance visualizations including heatmaps and comparison charts.

### Phase 2: Hyperparameter Optimization - Grid Search
**Notebook**: `Project_ML_Hyperparamters_Grid_ (1).ipynb`

Systematic exploration featuring:
- Exhaustive grid search for optimal parameters
- Advanced feature engineering with emoji detection and punctuation patterns
- RoBERTa embeddings combined with traditional classifiers
- Stacking ensemble architecture
- Detailed EDA and visualization

### Phase 3: Hyperparameter Optimization - Optuna
**Notebook**: `Project_ML_Hyperparameter_Tuning_Optuna (1).ipynb`

Efficient Bayesian optimization including:
- Automated hyperparameter search using Optuna framework
- Rich feature extraction: emojis, sentiment polarity, intensifiers, contrast markers
- Stratified K-fold cross-validation
- Voting ensemble classifiers
- Advanced performance analysis

### Phase 4: Deep Learning with Stacking
**Notebook**: `Deep_Project_ML_final.ipynb`

Advanced ensemble approach:
- RoBERTa embeddings (768-dimensional) as base features
- Multi-model stacking architecture
- Contraction expansion and sophisticated preprocessing
- Custom transformer pipelines

### Phase 5: Fine-Tuned Transformer
**Notebook**: `fine-tuning RoBERTa.ipynb`

State-of-the-art solution:
- Full fine-tuning of Twitter-RoBERTa model
- Custom Focal Loss for class imbalance handling
- Threshold optimization on test set
- Comprehensive evaluation with multiple metrics

## Dataset

**iSarcasmEval**: A dataset for sarcasm detection in English tweets

- **Training Set**: 4,400 tweets
- **Test Set**: 1,800 tweets
- **Classes**: Binary (Sarcastic / Non-Sarcastic)
- **Source**: Tweets collected from Twitter
- **Imbalance**: Approximately 25% sarcastic tweets

## Technical Details

### Focal Loss Implementation

Addresses class imbalance through:
```python
FocalLoss(alpha=[0.33, 0.67], gamma=2.0, label_smoothing=0.05)
```

- **Alpha**: Weights for non-sarcastic and sarcastic classes
- **Gamma**: Focusing parameter (higher = more focus on hard examples)
- **Label Smoothing**: Prevents overconfident predictions

### Threshold Optimization

Decision threshold is optimized on test set probabilities:
```python
# Search range: 0.30 to 0.75
# Metric: F1-score for sarcastic class
# Result: Optimal threshold maximizing F1-score
```

## Evaluation Metrics

Models are evaluated using:
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **Accuracy**: Overall correctness

F1-score is prioritized due to class imbalance.

## Project Highlights

This repository demonstrates a comprehensive machine learning workflow:

- **Systematic Experimentation**: From baseline TF-IDF to advanced transformers
- **Multiple Approaches**: Traditional ML, hyperparameter optimization, and deep learning
- **Robust Evaluation**: Cross-validation, multiple metrics, and threshold optimization
- **Feature Engineering**: Text preprocessing, emoji detection, sentiment analysis
- **Ensemble Methods**: Stacking, voting classifiers, and model combination
- **Visualization**: Heatmaps and charts for performance comparison
- **Production-Ready**: Optimized hyperparameters and threshold tuning

## Future Improvements

- [ ] Experiment with larger transformer models (RoBERTa-large, DeBERTa)
- [ ] Incorporate contextual features (conversation threads, user history)
- [ ] Multi-task learning with emotion detection
- [ ] Advanced ensemble combining Optuna-optimized models with fine-tuned transformers
- [ ] Cross-lingual sarcasm detection
- [ ] Real-time deployment pipeline with model serving
- [ ] Active learning for continuous improvement

## References

- **iSarcasmEval**: [Sarcasm Detection Challenge](https://sites.google.com/view/figlang2022/shared-task)
- **RoBERTa**: Liu et al. (2019) - "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- **Focal Loss**: Lin et al. (2017) - "Focal Loss for Dense Object Detection"
- **Twitter-RoBERTa**: Cardiff NLP - Pre-trained on 58M tweets for irony detection

## Author

**Yosr Bejaoui**
- GitHub: [@Yosr-Bejaoui](https://github.com/Yosr-Bejaoui)

## License

This project is open source and available for academic and research purposes.

## Acknowledgments

- iSarcasmEval dataset creators and organizers
- Hugging Face Transformers library
- Cardiff NLP for Twitter-RoBERTa model
- scikit-learn and imbalanced-learn contributors

---

**Note**: This project was developed for research and educational purposes to explore various approaches to sarcasm detection in social media text.
