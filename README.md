# Google IT Automation with Python Professional Certificate - Practice files

This repository contains the practice files used throughout the courses that are
part of the Google IT Automation with Python Professional Certificate

There's a separate folder for each course.

## 🆕 Logistic Regression Classifier for MNIST

A complete implementation of multi-class logistic regression for MNIST digit classification.

### Quick Start
```bash
pip install -r requirements.txt
python test_classifier.py  # Run tests
python mnist_train.py      # Train on MNIST
```

### Documentation
- **[ANSWER_TO_QUESTION.md](ANSWER_TO_QUESTION.md)** - How to fix low accuracy (~10% → ~85-92%)
- **[BEFORE_AND_AFTER.md](BEFORE_AND_AFTER.md)** - Side-by-side code comparison
- **[README_CLASSIFIER.md](README_CLASSIFIER.md)** - Complete usage guide
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Technical details of all fixes

### Key Fixes for Accuracy
1. ❌ **Don't normalize labels** (`y_train / 255` is wrong!)
2. ✅ **Do normalize test data** (`X_test / 255` required!)
3. ✅ **Use stable softmax** (subtract max before exp)

