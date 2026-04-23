# Machine Learning vs Deep Learning on MNIST

**A data-driven comparison: when does deep learning actually outperform traditional machine learning?**

## Key Finding

> **On large-scale data, deep learning achieves both higher accuracy AND faster training than traditional ML.**

This isn't just "DL is better". The key insight is the **data scale tipping point**:

| Data Scale | Winner (Accuracy) | Winner (Training Speed) |
|------------|-------------------|--------------------------|
| Small (e.g., 1k samples) | Traditional ML | Traditional ML |
| **Large (full MNIST, 60k samples)** | **Deep Learning** | **Deep Learning** |

**Takeaway**: As data scale increases, deep learning's advantage grows — both in accuracy and efficiency.

## Why This Matters

Many introductory courses teach that "deep learning is slow but accurate". This project shows that's only true for small datasets. **At scale, deep learning wins on both fronts.**

## How to Reproduce

```bash
git clone <your-repo>
pip install -r requirements.txt
python mnist_comparison.py
```

The script will automatically run all models and print the comparison table.


## Next Steps

This finding led me to explore more complex sequence tasks. See my other project:  
**[RNN for IMDb Sentiment Classification]** — where deep learning's advantage becomes even more pronounced.

---

*Part of my self-learning journey in AI. Always open to feedback.*
