# Portfolio

A collection of machine learning and data science projects showcasing research implementations and engineering work.

---

## Projects

### 🧠 Neural Networks

#### [GCM — Graph Convolution Machine for Context-Aware Recommendation](./neural-networks/gcm-tensorflow-v2)

A **Graph Convolution Machine (GCM)** implementation using the TensorFlow 2 Keras API (on TensorFlow-GPU 1.15), a context-aware recommender system that leverages Graph Neural Networks to model user–item–context interactions.

**Paper:** Jiancan Wu, Xiangnan He, Xiang Wang, et al. *Graph Convolution Machine for Context-aware Recommender System.* arXiv:2001.11402 (2020). [[Link]](https://arxiv.org/abs/2001.11402)

**Key highlights:**
- Fixes execution issues present in the original source code, migrating it to the TensorFlow 2 Keras API while running on TensorFlow-GPU 1.15.
- Supports multiple decoder types: **FM**, **FM-Pooling**, and **Inner Product (IP)**.
- Implements GCN-based user/item/context encoding with configurable graph normalisation strategies (`ls`, `rs`, `rd`, `db`).
- Trains with either **log loss** (pointwise) or **BPR loss** (pairwise).
- Benchmarked on **Yelp-NC**, **Yelp-OH**, and **Amazon-Book** datasets.

**Tech stack:** Python 3.8 · TensorFlow-GPU 1.15 (TF2 API) · NumPy · SciPy · Pandas · Cython

**Quick start:**
```bash
# Install dependencies
pip install -r neural-networks/gcm-tensorflow-v2/requirements.txt

# Compile the C++ evaluator (optional but recommended for speed)
cd neural-networks/gcm-tensorflow-v2
python setup.py build_ext --inplace

# Train on Yelp-NC
python GCM.py --dataset Yelp-NC --num_gcn_layers 2 --reg 1e-3 \
              --decoder_type FM --adj_norm_type ls --num_negatives 4
```

---

## Repository Structure

```
portfolio/
└── neural-networks/
    └── gcm-tensorflow-v2/   # GCM context-aware recommender (TF2)
```

---

## Contact

Feel free to reach out or connect if you have questions about any of the projects above.
