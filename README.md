# NLP Homework 04 – Neural Networks

Short, self‑contained project with two parts:

1. **Custom autograd ops**  
   - `MyMul`, `MyMax` (element‑wise)  
   - `CosLinear` (weights wrapped in `cos`)

2. **Dynamic Network**  
   - Randomly applies hidden layer 1 – 4 times during training  
   - Averages 1 – 4 times during evaluation  
   - 4‑fold CV (75 % / 25 %) on a numpy regression set (2 ** 14 samples)

---

## 🔧 Quick start

```bash
# create / activate your virtual env first

# install dependencies from pyproject.toml
uv sync

# run the homework script 
uv run NLP_04.py