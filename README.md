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
│   ├── train.En.csv              # Training dataset
│   └── task_A_En_test.csv        # Test dataset
├── Model_Exploration.ipynb        # Comprehensive model comparison
├── fine-tuning RoBERTa.ipynb     # Fine-tuned RoBERTa implementation
├── comprehensive_model_results.csv # All model evaluation results
└── README.md                      # Project documentation
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

### 5. Fine-Tuned RoBERTa Model

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

## Getting Started

### Prerequisites

```bash
python >= 3.8
torch >= 1.9.0
transformers >= 4.20.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
xgboost >= 1.5.0
imbalanced-learn >= 0.9.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Yosr-Bejaoui/sarcasm-detector.git
cd sarcasm-detector
```

2. Install dependencies:
```bash
pip install torch transformers scikit-learn pandas numpy xgboost imbalanced-learn gdown
```

3. Download the dataset (automatically done in notebooks):
```python
import gdown
file_id_train = '1x6CbYlfuPZf1-EZFVN-uKcFptlthVGf8'
gdown.download(f'https://drive.google.com/uc?id={file_id_train}', 'train.En.csv', quiet=False)
```

### Usage

#### Model Exploration

Run the comprehensive model comparison:
```bash
jupyter notebook "Model_Exploration.ipynb"
```

This notebook will:
- Extract features using TF-IDF and transformer models
- Apply various sampling techniques
- Train and evaluate all model combinations
- Generate comparison visualizations

#### Fine-Tuned RoBERTa

Train the optimized RoBERTa model:
```bash
jupyter notebook "fine-tuning RoBERTa.ipynb"
```

This notebook will:
- Load and preprocess the iSarcasmEval dataset
- Fine-tune Twitter-RoBERTa with Focal Loss
- Optimize classification threshold
- Evaluate on test set

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

## Future Improvements

- [ ] Experiment with larger transformer models (RoBERTa-large, DeBERTa)
- [ ] Incorporate contextual features (conversation threads, user history)
- [ ] Multi-task learning with emotion detection
- [ ] Ensemble methods combining top-performing models
- [ ] Cross-lingual sarcasm detection
- [ ] Real-time deployment pipeline

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
