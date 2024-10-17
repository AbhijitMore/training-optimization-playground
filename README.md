# 🧑‍💻 Training Optimization Playground 🎯

Welcome to **Training Optimization Playground**! This repository is a fun and interactive space where we explore various **training optimizations** for neural networks. Whether you’re curious about how different optimizers perform or how to find the optimal learning rate for your models, this playground has something for you!

## 🚀 Project Overview
In this repository, we cover:
- 📊 **Optimizer Comparison**: Compare different optimizers like `SGD`, `Adam`, and `RMSprop`.
- 🔍 **Learning Rate Finder**: Automatically find the best learning rate to speed up training without sacrificing accuracy.
- Detailed performance metrics are plotted for easy comparison of experiments.

This repository is divided into two parts:
1. **Optimizer Comparison** (`optimizer_comparison/`): Experimenting with multiple optimizers and analyzing their performance.
2. **Learning Rate Finder** (`find_optimal_lr/`): Finding the optimal learning rate for better training.

---

## 📂 Directory Structure

```
training-optimization-playground/
├── optimizer_comparison/
│   ├── resources/
│   │   ├── console_output.png  # Sample console output from optimizer comparison
│   │   ├── metrics.png         # Metrics comparison between different learning rates
│   ├── utils.py                # Utility functions for loading data, plotting metrics
│   ├── nets.py                 # Simple neural network model definitions
│   ├── main.py                 # Main script to run optimizer comparison experiments
├── find_optimal_lr/
│   ├── resources/
│   │   ├── learning_rate.png    # LR Finder graph
│   │   ├── console_output.png   # Sample console output from LR finding process
│   │   ├── metrics_comparison.png # Metrics comparison after applying optimal learning rate
│   ├── find_lr.py               # Script for finding optimal learning rate
│   ├── main.py                  # Main script to demonstrate training with optimal LR
```

---

## 🛠️ Usage

### Optimizer Comparison

Run the **optimizer comparison** experiment to see how `SGD`, `Adam`, and `RMSprop` perform with your dataset.

```bash
cd optimizer_comparison/
python main.py
```

The **console output** and **metrics plots** will be saved to the `resources/` directory.

![Optimizer Comparison Metrics](optimizer_comparison/resources/metrics.png)

---

### Learning Rate Finder

Run the **learning rate finder** to determine the optimal learning rate for your model.

```bash
cd find_optimal_lr/
python main.py
```

The **learning rate plot** and **metrics comparison** will be saved to the `resources/` directory.

![Learning Rate Finder](find_optimal_lr/resources/learning_rate.png)

---

## 📈 Results and Visualization

- **Optimizer Comparison**: We experiment with different optimizers and visualize their performance. Check out the plots in the `resources/` folder:
  ![Metrics](optimizer_comparison/resources/metrics.png)

- **Learning Rate Finder**: Visualize the learning rate vs. loss graph to pinpoint the best learning rate:
  ![Learning Rate Plot](find_optimal_lr/resources/learning_rate.png)

---

The accuracy and loss plots give a visual summary of model performance across epochs, indicating improvements and stability.

## Contributing
Contributions are welcome! 🎉 Whether you're reporting a bug, suggesting a new feature, or submitting a pull request, your input is valuable.