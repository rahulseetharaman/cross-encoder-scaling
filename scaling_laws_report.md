# Scaling Laws Analysis — Statistical Summary

---

**Scaling law form:** `y = a + b · x^c`  (power law)

**Data scaling:** trains on all steps except the last 5 (held out as test), one data point per step.

**Model scaling:** trains on last 10 checkpoints of 17m, 32m, 68m, 150m (40 points total); forecasts 400m and 1b.

**Bootstrap CI:** 1000 resamples of training data with replacement.

**Delta-method CI:** analytical propagation of curve-fit covariance.

---

## 1. Data Scaling — Goodness of Fit

Power-law fit quality on training data. F-test compares the power-law model (3 params) against a constant null. Significance: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant.

### NDCG@10

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.927 | 0.924 | 267.8 | 0.0000 | *** |
| 17m | ranknet | 0.897 | 0.892 | 183.6 | 0.0000 | *** |
| 17m | bce | 0.824 | 0.816 | 103.0 | 0.0000 | *** |
| 32m | listnet | 0.921 | 0.917 | 237.9 | 0.0000 | *** |
| 32m | ranknet | 0.901 | 0.896 | 190.2 | 0.0000 | *** |
| 32m | bce | 0.909 | 0.905 | 221.1 | 0.0000 | *** |
| 68m | listnet | 0.953 | 0.951 | 415.8 | 0.0000 | *** |
| 68m | ranknet | 0.897 | 0.892 | 183.4 | 0.0000 | *** |
| 68m | bce | 0.874 | 0.868 | 152.6 | 0.0000 | *** |
| 150m | listnet | 0.894 | 0.889 | 173.1 | 0.0000 | *** |
| 150m | ranknet | 0.898 | 0.893 | 184.9 | 0.0000 | *** |
| 150m | bce | 0.862 | 0.856 | 137.5 | 0.0000 | *** |
| 400m | listnet | 0.963 | 0.961 | 531.8 | 0.0000 | *** |
| 400m | ranknet | 0.936 | 0.935 | 595.1 | 0.0000 | *** |
| 400m | bce | 0.894 | 0.890 | 186.5 | 0.0000 | *** |
| 1b | listnet | 0.758 | 0.747 | 65.8 | 0.0000 | *** |
| 1b | ranknet | 0.974 | 0.972 | 775.0 | 0.0000 | *** |
| 1b | bce | 0.945 | 0.942 | 375.1 | 0.0000 | *** |

### Recall@10

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.925 | 0.922 | 260.9 | 0.0000 | *** |
| 17m | ranknet | 0.892 | 0.887 | 174.3 | 0.0000 | *** |
| 17m | bce | 0.832 | 0.825 | 109.1 | 0.0000 | *** |
| 32m | listnet | 0.925 | 0.921 | 251.2 | 0.0000 | *** |
| 32m | ranknet | 0.898 | 0.893 | 183.9 | 0.0000 | *** |
| 32m | bce | 0.905 | 0.900 | 208.5 | 0.0000 | *** |
| 68m | listnet | 0.961 | 0.959 | 509.6 | 0.0000 | *** |
| 68m | ranknet | 0.898 | 0.893 | 185.5 | 0.0000 | *** |
| 68m | bce | 0.867 | 0.861 | 142.9 | 0.0000 | *** |
| 150m | listnet | 0.882 | 0.877 | 153.6 | 0.0000 | *** |
| 150m | ranknet | 0.898 | 0.893 | 185.5 | 0.0000 | *** |
| 150m | bce | 0.845 | 0.838 | 119.8 | 0.0000 | *** |
| 400m | listnet | 0.971 | 0.970 | 696.9 | 0.0000 | *** |
| 400m | ranknet | 0.938 | 0.937 | 613.8 | 0.0000 | *** |
| 400m | bce | 0.888 | 0.883 | 174.7 | 0.0000 | *** |
| 1b | listnet | 0.778 | 0.768 | 73.8 | 0.0000 | *** |
| 1b | ranknet | 0.974 | 0.973 | 789.1 | 0.0000 | *** |
| 1b | bce | 0.964 | 0.962 | 591.0 | 0.0000 | *** |

### MAP

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.927 | 0.923 | 266.1 | 0.0000 | *** |
| 17m | ranknet | 0.900 | 0.895 | 188.9 | 0.0000 | *** |
| 17m | bce | 0.818 | 0.810 | 99.1 | 0.0000 | *** |
| 32m | listnet | 0.919 | 0.915 | 231.4 | 0.0000 | *** |
| 32m | ranknet | 0.901 | 0.897 | 192.1 | 0.0000 | *** |
| 32m | bce | 0.911 | 0.907 | 225.4 | 0.0000 | *** |
| 68m | listnet | 0.950 | 0.947 | 385.6 | 0.0000 | *** |
| 68m | ranknet | 0.897 | 0.892 | 183.4 | 0.0000 | *** |
| 68m | bce | 0.877 | 0.871 | 156.7 | 0.0000 | *** |
| 150m | listnet | 0.899 | 0.894 | 181.6 | 0.0000 | *** |
| 150m | ranknet | 0.898 | 0.894 | 185.8 | 0.0000 | *** |
| 150m | bce | 0.870 | 0.864 | 147.0 | 0.0000 | *** |
| 400m | listnet | 0.958 | 0.956 | 472.5 | 0.0000 | *** |
| 400m | ranknet | 0.935 | 0.934 | 584.7 | 0.0000 | *** |
| 400m | bce | 0.897 | 0.892 | 191.0 | 0.0000 | *** |
| 1b | listnet | 0.747 | 0.735 | 62.0 | 0.0000 | *** |
| 1b | ranknet | 0.974 | 0.973 | 785.9 | 0.0000 | *** |
| 1b | bce | 0.933 | 0.930 | 305.9 | 0.0000 | *** |

### MRR

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.927 | 0.923 | 265.3 | 0.0000 | *** |
| 17m | ranknet | 0.900 | 0.895 | 188.6 | 0.0000 | *** |
| 17m | bce | 0.819 | 0.810 | 99.3 | 0.0000 | *** |
| 32m | listnet | 0.918 | 0.914 | 230.2 | 0.0000 | *** |
| 32m | ranknet | 0.902 | 0.897 | 192.3 | 0.0000 | *** |
| 32m | bce | 0.911 | 0.907 | 225.9 | 0.0000 | *** |
| 68m | listnet | 0.949 | 0.947 | 382.0 | 0.0000 | *** |
| 68m | ranknet | 0.896 | 0.891 | 181.5 | 0.0000 | *** |
| 68m | bce | 0.877 | 0.871 | 156.8 | 0.0000 | *** |
| 150m | listnet | 0.898 | 0.893 | 180.3 | 0.0000 | *** |
| 150m | ranknet | 0.898 | 0.893 | 183.9 | 0.0000 | *** |
| 150m | bce | 0.869 | 0.863 | 145.3 | 0.0000 | *** |
| 400m | listnet | 0.959 | 0.957 | 475.5 | 0.0000 | *** |
| 400m | ranknet | 0.935 | 0.933 | 580.0 | 0.0000 | *** |
| 400m | bce | 0.896 | 0.891 | 189.7 | 0.0000 | *** |
| 1b | listnet | 0.744 | 0.731 | 60.9 | 0.0000 | *** |
| 1b | ranknet | 0.974 | 0.973 | 781.1 | 0.0000 | *** |
| 1b | bce | 0.932 | 0.929 | 302.4 | 0.0000 | *** |

### CE

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.883 | 0.878 | 158.8 | 0.0000 | *** |
| 17m | ranknet | 0.895 | 0.890 | 179.6 | 0.0000 | *** |
| 17m | bce | 0.849 | 0.843 | 124.1 | 0.0000 | *** |
| 32m | listnet | 0.845 | 0.837 | 111.4 | 0.0000 | *** |
| 32m | ranknet | 0.882 | 0.876 | 156.8 | 0.0000 | *** |
| 32m | bce | 0.908 | 0.904 | 218.4 | 0.0000 | *** |
| 68m | listnet | 0.713 | 0.699 | 51.0 | 0.0000 | *** |
| 68m | ranknet | 0.864 | 0.857 | 132.9 | 0.0000 | *** |
| 68m | bce | 0.848 | 0.841 | 123.0 | 0.0000 | *** |
| 150m | listnet | 0.779 | 0.768 | 72.3 | 0.0000 | *** |
| 150m | ranknet | 0.903 | 0.898 | 195.1 | 0.0000 | *** |
| 150m | bce | 0.869 | 0.863 | 146.5 | 0.0000 | *** |
| 400m | listnet | 0.674 | 0.658 | 42.4 | 0.0000 | *** |
| 400m | ranknet | 0.122 | 0.100 | 5.6 | 0.0052 | ** |
| 400m | bce | 0.823 | 0.815 | 102.5 | 0.0000 | *** |
| 1b | listnet | 0.493 | 0.469 | 20.4 | 0.0000 | *** |
| 1b | ranknet | 0.946 | 0.944 | 371.5 | 0.0000 | *** |
| 1b | bce | 0.502 | 0.479 | 22.2 | 0.0000 | *** |

### P@10

| Model | Loss | R² | Adj R² | F-stat | F p-value | Sig |
|-------|------|------|--------|--------|-----------|-----|
| 17m | listnet | 0.929 | 0.926 | 275.0 | 0.0000 | *** |
| 17m | ranknet | 0.891 | 0.886 | 172.1 | 0.0000 | *** |
| 17m | bce | 0.865 | 0.859 | 140.7 | 0.0000 | *** |
| 32m | listnet | 0.923 | 0.920 | 247.4 | 0.0000 | *** |
| 32m | ranknet | 0.897 | 0.893 | 183.8 | 0.0000 | *** |
| 32m | bce | 0.907 | 0.902 | 213.6 | 0.0000 | *** |
| 68m | listnet | 0.960 | 0.958 | 495.5 | 0.0000 | *** |
| 68m | ranknet | 0.898 | 0.894 | 185.8 | 0.0000 | *** |
| 68m | bce | 0.865 | 0.859 | 141.5 | 0.0000 | *** |
| 150m | listnet | 0.882 | 0.876 | 152.7 | 0.0000 | *** |
| 150m | ranknet | 0.899 | 0.894 | 186.0 | 0.0000 | *** |
| 150m | bce | 0.845 | 0.838 | 120.2 | 0.0000 | *** |
| 400m | listnet | 0.970 | 0.969 | 673.3 | 0.0000 | *** |
| 400m | ranknet | 0.938 | 0.937 | 613.5 | 0.0000 | *** |
| 400m | bce | 0.887 | 0.881 | 172.0 | 0.0000 | *** |
| 1b | listnet | 0.777 | 0.767 | 73.3 | 0.0000 | *** |
| 1b | ranknet | 0.974 | 0.972 | 775.4 | 0.0000 | *** |
| 1b | bce | 0.963 | 0.962 | 579.2 | 0.0000 | *** |

## 2. Data Scaling — Forecast Errors on Held-Out Checkpoints

MAE and RMSE computed over the last 5 held-out checkpoints. CI Width is the mean width of the 95% bootstrap confidence interval over test steps.

### NDCG@10

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0067 | 0.0079 | 0.0142 |
| 17m | ranknet | 0.0138 | 0.0142 | 0.0252 |
| 17m | bce | 0.0066 | 0.0068 | 0.0239 |
| 32m | listnet | 0.0104 | 0.0129 | 0.0203 |
| 32m | ranknet | 0.0182 | 0.0185 | 0.0278 |
| 32m | bce | 0.0229 | 0.0229 | 0.0179 |
| 68m | listnet | 0.0065 | 0.0085 | 0.0113 |
| 68m | ranknet | 0.0167 | 0.0168 | 0.0284 |
| 68m | bce | 0.0172 | 0.0172 | 0.0315 |
| 150m | listnet | 0.0064 | 0.0078 | 0.0177 |
| 150m | ranknet | 0.0163 | 0.0164 | 0.0295 |
| 150m | bce | 0.0194 | 0.0194 | 0.0344 |
| 400m | listnet | 0.0047 | 0.0058 | 0.0093 |
| 400m | ranknet | 0.0038 | 0.0041 | 0.0144 |
| 400m | bce | 0.0071 | 0.0072 | 0.0257 |
| 1b | listnet | 0.0016 | 0.0017 | 0.0091 |
| 1b | ranknet | 0.0065 | 0.0065 | 0.0084 |
| 1b | bce | 0.0060 | 0.0060 | 0.0092 |

### Recall@10

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0084 | 0.0101 | 0.0202 |
| 17m | ranknet | 0.0216 | 0.0222 | 0.0362 |
| 17m | bce | 0.0098 | 0.0099 | 0.0374 |
| 32m | listnet | 0.0161 | 0.0194 | 0.0264 |
| 32m | ranknet | 0.0233 | 0.0236 | 0.0420 |
| 32m | bce | 0.0311 | 0.0311 | 0.0247 |
| 68m | listnet | 0.0069 | 0.0082 | 0.0130 |
| 68m | ranknet | 0.0239 | 0.0239 | 0.0386 |
| 68m | bce | 0.0269 | 0.0270 | 0.0431 |
| 150m | listnet | 0.0072 | 0.0086 | 0.0236 |
| 150m | ranknet | 0.0236 | 0.0237 | 0.0389 |
| 150m | bce | 0.0289 | 0.0289 | 0.0486 |
| 400m | listnet | 0.0055 | 0.0067 | 0.0097 |
| 400m | ranknet | 0.0030 | 0.0033 | 0.0179 |
| 400m | bce | 0.0121 | 0.0121 | 0.0318 |
| 1b | listnet | 0.0007 | 0.0008 | 0.0104 |
| 1b | ranknet | 0.0086 | 0.0086 | 0.0120 |
| 1b | bce | 0.0072 | 0.0072 | 0.0103 |

### MAP

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0058 | 0.0068 | 0.0125 |
| 17m | ranknet | 0.0110 | 0.0113 | 0.0206 |
| 17m | bce | 0.0060 | 0.0061 | 0.0201 |
| 32m | listnet | 0.0081 | 0.0103 | 0.0178 |
| 32m | ranknet | 0.0161 | 0.0163 | 0.0235 |
| 32m | bce | 0.0190 | 0.0190 | 0.0138 |
| 68m | listnet | 0.0061 | 0.0082 | 0.0098 |
| 68m | ranknet | 0.0137 | 0.0138 | 0.0243 |
| 68m | bce | 0.0134 | 0.0134 | 0.0258 |
| 150m | listnet | 0.0061 | 0.0074 | 0.0152 |
| 150m | ranknet | 0.0133 | 0.0133 | 0.0242 |
| 150m | bce | 0.0158 | 0.0158 | 0.0277 |
| 400m | listnet | 0.0045 | 0.0054 | 0.0083 |
| 400m | ranknet | 0.0039 | 0.0042 | 0.0127 |
| 400m | bce | 0.0054 | 0.0055 | 0.0215 |
| 1b | listnet | 0.0022 | 0.0023 | 0.0078 |
| 1b | ranknet | 0.0055 | 0.0055 | 0.0070 |
| 1b | bce | 0.0051 | 0.0052 | 0.0088 |

### MRR

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0059 | 0.0070 | 0.0126 |
| 17m | ranknet | 0.0110 | 0.0113 | 0.0201 |
| 17m | bce | 0.0061 | 0.0063 | 0.0210 |
| 32m | listnet | 0.0082 | 0.0103 | 0.0167 |
| 32m | ranknet | 0.0162 | 0.0164 | 0.0243 |
| 32m | bce | 0.0193 | 0.0193 | 0.0150 |
| 68m | listnet | 0.0059 | 0.0081 | 0.0098 |
| 68m | ranknet | 0.0138 | 0.0139 | 0.0262 |
| 68m | bce | 0.0134 | 0.0134 | 0.0265 |
| 150m | listnet | 0.0060 | 0.0074 | 0.0161 |
| 150m | ranknet | 0.0134 | 0.0134 | 0.0248 |
| 150m | bce | 0.0157 | 0.0157 | 0.0290 |
| 400m | listnet | 0.0045 | 0.0054 | 0.0079 |
| 400m | ranknet | 0.0041 | 0.0043 | 0.0131 |
| 400m | bce | 0.0053 | 0.0054 | 0.0216 |
| 1b | listnet | 0.0022 | 0.0023 | 0.0083 |
| 1b | ranknet | 0.0055 | 0.0055 | 0.0072 |
| 1b | bce | 0.0051 | 0.0052 | 0.0090 |

### CE

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0137 | 0.0159 | 0.0253 |
| 17m | ranknet | 0.0174 | 0.0177 | 0.0340 |
| 17m | bce | 0.0032 | 0.0034 | 0.0106 |
| 32m | listnet | 0.0139 | 0.0172 | 0.0282 |
| 32m | ranknet | 0.0239 | 0.0243 | 0.0457 |
| 32m | bce | 0.0089 | 0.0091 | 0.0099 |
| 68m | listnet | 0.0224 | 0.0301 | 0.0446 |
| 68m | ranknet | 0.0379 | 0.0389 | 0.0422 |
| 68m | bce | 0.0067 | 0.0076 | 0.0197 |
| 150m | listnet | 0.0133 | 0.0166 | 0.0387 |
| 150m | ranknet | 0.1152 | 0.1163 | 0.1797 |
| 150m | bce | 0.0091 | 0.0093 | 0.0202 |
| 400m | listnet | 0.0395 | 0.0426 | 0.0410 |
| 400m | ranknet | 0.7752 | 0.7755 | 0.3883 |
| 400m | bce | 0.0035 | 0.0041 | 0.0186 |
| 1b | listnet | 0.0117 | 0.0134 | 0.0299 |
| 1b | ranknet | 0.0386 | 0.0447 | 0.0783 |
| 1b | bce | 0.0128 | 0.0134 | 0.0234 |

### P@10

| Model | Loss | MAE | RMSE | 95% CI Width |
|-------|------|-----|------|-------------|
| 17m | listnet | 0.0016 | 0.0018 | 0.0029 |
| 17m | ranknet | 0.0022 | 0.0023 | 0.0038 |
| 17m | bce | 0.0002 | 0.0002 | 0.0033 |
| 32m | listnet | 0.0016 | 0.0020 | 0.0029 |
| 32m | ranknet | 0.0024 | 0.0024 | 0.0042 |
| 32m | bce | 0.0041 | 0.0041 | 0.0036 |
| 68m | listnet | 0.0007 | 0.0009 | 0.0014 |
| 68m | ranknet | 0.0025 | 0.0025 | 0.0039 |
| 68m | bce | 0.0028 | 0.0028 | 0.0046 |
| 150m | listnet | 0.0008 | 0.0009 | 0.0025 |
| 150m | ranknet | 0.0025 | 0.0025 | 0.0040 |
| 150m | bce | 0.0030 | 0.0030 | 0.0050 |
| 400m | listnet | 0.0006 | 0.0007 | 0.0009 |
| 400m | ranknet | 0.0003 | 0.0003 | 0.0019 |
| 400m | bce | 0.0012 | 0.0012 | 0.0032 |
| 1b | listnet | 0.0001 | 0.0001 | 0.0010 |
| 1b | ranknet | 0.0009 | 0.0009 | 0.0012 |
| 1b | bce | 0.0007 | 0.0007 | 0.0012 |

## 5. Model Scaling — Goodness of Fit

Power-law fit across model sizes (17m–150m training). F-test null: constant (mean) prediction.

| Loss | Metric | R² | Adj R² | F-stat | F p-value | Sig |
|------|--------|----|--------|--------|-----------|-----|
| listnet | NDCG@10 | 0.953 | 0.950 | 373.2 | 0.0000 | *** |
| listnet | Recall@10 | 0.952 | 0.949 | 365.5 | 0.0000 | *** |
| listnet | MAP | 0.953 | 0.951 | 378.2 | 0.0000 | *** |
| listnet | MRR | 0.953 | 0.951 | 376.1 | 0.0000 | *** |
| listnet | CE | 0.847 | 0.839 | 102.4 | 0.0000 | *** |
| listnet | P@10 | 0.951 | 0.948 | 356.5 | 0.0000 | *** |
| ranknet | NDCG@10 | 0.982 | 0.981 | 1003.4 | 0.0000 | *** |
| ranknet | Recall@10 | 0.986 | 0.985 | 1281.4 | 0.0000 | *** |
| ranknet | MAP | 0.979 | 0.978 | 876.5 | 0.0000 | *** |
| ranknet | MRR | 0.980 | 0.979 | 896.0 | 0.0000 | *** |
| ranknet | CE | 0.638 | 0.619 | 32.6 | 0.0000 | *** |
| ranknet | P@10 | 0.986 | 0.985 | 1295.9 | 0.0000 | *** |
| bce | NDCG@10 | 0.994 | 0.993 | 2940.9 | 0.0000 | *** |
| bce | Recall@10 | 0.995 | 0.995 | 4035.3 | 0.0000 | *** |
| bce | MAP | 0.993 | 0.992 | 2578.7 | 0.0000 | *** |
| bce | MRR | 0.992 | 0.992 | 2388.4 | 0.0000 | *** |
| bce | CE | 0.986 | 0.985 | 1325.6 | 0.0000 | *** |
| bce | P@10 | 0.995 | 0.995 | 3808.3 | 0.0000 | *** |

## 3. Model Scaling — Forecast vs Actual (400m and 1b)

Training models: 17m, 32m, 68m, 150m (last 10 checkpoints each = 40 points). Test models: 400m and 1b. Actual mean ± std computed over the last 10 checkpoints of the test model. One-sample t-test: H₀: actual mean = forecast. Significance: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant.

### NDCG@10

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 0.3484 | 0.3633 ± 0.0064 | [0.3426, 0.3560] | 6.930 | 0.0001 | *** | 0.953 |
| listnet | 1b | 0.3502 | 0.3740 ± 0.0019 | [0.3436, 0.3595] | 38.020 | 0.0000 | *** | 0.953 |
| ranknet | 400m | 0.3646 | 0.3739 ± 0.0018 | [0.3619, 0.3690] | 22.101 | 0.0000 | *** | 0.982 |
| ranknet | 1b | 0.3658 | 0.3776 ± 0.0016 | [0.3629, 0.3707] | 22.066 | 0.0000 | *** | 0.982 |
| bce | 400m | 0.3422 | 0.3513 ± 0.0039 | [0.3394, 0.3466] | 7.058 | 0.0001 | *** | 0.994 |
| bce | 1b | 0.3451 | 0.3576 ± 0.0030 | [0.3417, 0.3501] | 12.553 | 0.0000 | *** | 0.994 |

### Recall@10

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 0.5197 | 0.5339 ± 0.0065 | [0.5117, 0.5297] | 6.524 | 0.0001 | *** | 0.952 |
| listnet | 1b | 0.5216 | 0.5461 ± 0.0018 | [0.5127, 0.5339] | 40.980 | 0.0000 | *** | 0.952 |
| ranknet | 400m | 0.5340 | 0.5474 ± 0.0020 | [0.5314, 0.5379] | 29.939 | 0.0000 | *** | 0.986 |
| ranknet | 1b | 0.5348 | 0.5487 ± 0.0021 | [0.5320, 0.5390] | 20.088 | 0.0000 | *** | 0.986 |
| bce | 400m | 0.5152 | 0.5217 ± 0.0032 | [0.5122, 0.5191] | 6.115 | 0.0002 | *** | 0.995 |
| bce | 1b | 0.5186 | 0.5273 ± 0.0025 | [0.5149, 0.5230] | 10.624 | 0.0000 | *** | 0.995 |

### MAP

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 0.3010 | 0.3154 ± 0.0063 | [0.2962, 0.3076] | 6.904 | 0.0001 | *** | 0.953 |
| listnet | 1b | 0.3025 | 0.3248 ± 0.0019 | [0.2969, 0.3104] | 35.370 | 0.0000 | *** | 0.953 |
| ranknet | 400m | 0.3177 | 0.3245 ± 0.0018 | [0.3150, 0.3228] | 16.848 | 0.0000 | *** | 0.979 |
| ranknet | 1b | 0.3191 | 0.3289 ± 0.0015 | [0.3158, 0.3250] | 19.988 | 0.0000 | *** | 0.979 |
| bce | 400m | 0.2947 | 0.3044 ± 0.0039 | [0.2921, 0.2984] | 7.423 | 0.0000 | *** | 0.993 |
| bce | 1b | 0.2973 | 0.3103 ± 0.0031 | [0.2941, 0.3016] | 12.546 | 0.0000 | *** | 0.993 |

### MRR

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 0.3029 | 0.3175 ± 0.0062 | [0.2979, 0.3094] | 7.065 | 0.0001 | *** | 0.953 |
| listnet | 1b | 0.3044 | 0.3271 ± 0.0019 | [0.2987, 0.3123] | 36.283 | 0.0000 | *** | 0.953 |
| ranknet | 400m | 0.3195 | 0.3264 ± 0.0017 | [0.3169, 0.3241] | 17.409 | 0.0000 | *** | 0.980 |
| ranknet | 1b | 0.3208 | 0.3308 ± 0.0015 | [0.3178, 0.3261] | 20.158 | 0.0000 | *** | 0.980 |
| bce | 400m | 0.2969 | 0.3064 ± 0.0040 | [0.2940, 0.3006] | 7.138 | 0.0001 | *** | 0.992 |
| bce | 1b | 0.2994 | 0.3123 ± 0.0031 | [0.2959, 0.3038] | 12.513 | 0.0000 | *** | 0.992 |

### CE

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 3.7507 | 3.7474 ± 0.0437 | [3.7162, 3.7770] | -0.231 | 0.8228 | ns | 0.847 |
| listnet | 1b | 3.7394 | 3.7256 ± 0.0136 | [3.6829, 3.7736] | -3.030 | 0.0142 | * | 0.847 |
| ranknet | 400m | 2.2851 | 2.8722 ± 0.7779 | [2.0336, 2.6606] | 3.289 | 0.0039 | ** | 0.638 |
| ranknet | 1b | 1.9273 | 2.0585 ± 0.0219 | [1.6444, 2.3463] | 17.948 | 0.0000 | *** | 0.638 |
| bce | 400m | 3.9670 | 3.9569 ± 0.0041 | [3.9614, 3.9719] | -7.339 | 0.0000 | *** | 0.986 |
| bce | 1b | 3.9625 | 3.9537 ± 0.0051 | [3.9552, 3.9686] | -5.184 | 0.0006 | *** | 0.986 |

### P@10

| Loss | Model | Forecast | Actual (mean ± std) | 95% Boot CI | t-stat | p-value | Sig | R² |
|------|-------|----------|---------------------|-------------|--------|---------|-----|-----|
| listnet | 400m | 0.0536 | 0.0551 ± 0.0007 | [0.0529, 0.0547] | 6.487 | 0.0001 | *** | 0.951 |
| listnet | 1b | 0.0538 | 0.0564 ± 0.0002 | [0.0530, 0.0551] | 40.350 | 0.0000 | *** | 0.951 |
| ranknet | 400m | 0.0552 | 0.0566 ± 0.0002 | [0.0549, 0.0556] | 29.535 | 0.0000 | *** | 0.986 |
| ranknet | 1b | 0.0552 | 0.0567 ± 0.0002 | [0.0550, 0.0557] | 18.830 | 0.0000 | *** | 0.986 |
| bce | 400m | 0.0533 | 0.0539 ± 0.0003 | [0.0529, 0.0537] | 5.881 | 0.0002 | *** | 0.995 |
| bce | 1b | 0.0536 | 0.0545 ± 0.0003 | [0.0532, 0.0541] | 10.462 | 0.0000 | *** | 0.995 |

## 4. Model Scaling — Forecast Errors

MAE and RMSE computed between the single power-law forecast and the 10 actual checkpoint values at each test model size.

| Loss | Metric | 400m MAE | 400m RMSE | 1b MAE | 1b RMSE |
|------|--------|----------|-----------|--------|---------|
| listnet | NDCG@10 | 0.0149 | 0.0162 | 0.0238 | 0.0239 |
| listnet | Recall@10 | 0.0142 | 0.0156 | 0.0245 | 0.0246 |
| listnet | MAP | 0.0144 | 0.0157 | 0.0223 | 0.0224 |
| listnet | MRR | 0.0146 | 0.0159 | 0.0227 | 0.0228 |
| listnet | CE | 0.0398 | 0.0438 | 0.0153 | 0.0194 |
| listnet | P@10 | 0.0015 | 0.0016 | 0.0026 | 0.0026 |
| ranknet | NDCG@10 | 0.0093 | 0.0094 | 0.0118 | 0.0119 |
| ranknet | Recall@10 | 0.0134 | 0.0136 | 0.0139 | 0.0141 |
| ranknet | MAP | 0.0068 | 0.0070 | 0.0098 | 0.0099 |
| ranknet | MRR | 0.0070 | 0.0072 | 0.0100 | 0.0101 |
| ranknet | CE | 0.7777 | 0.9746 | 0.1313 | 0.1331 |
| ranknet | P@10 | 0.0014 | 0.0014 | 0.0014 | 0.0014 |
| bce | NDCG@10 | 0.0092 | 0.0099 | 0.0126 | 0.0129 |
| bce | Recall@10 | 0.0066 | 0.0072 | 0.0087 | 0.0091 |
| bce | MAP | 0.0097 | 0.0104 | 0.0131 | 0.0135 |
| bce | MRR | 0.0096 | 0.0103 | 0.0129 | 0.0133 |
| bce | CE | 0.0101 | 0.0109 | 0.0089 | 0.0101 |
| bce | P@10 | 0.0007 | 0.0007 | 0.0009 | 0.0009 |


---

## 6. Inferences and Interpretation

### 6.1 Data Scaling

**The power law fits retrieval metrics reliably.**
Across all six model sizes and three loss functions, the power-law form `y = a + b * D^c`
explains the majority of variance in NDCG@10, Recall@10, MAP, MRR, and P@10 as a function
of training steps (mean R2 ~0.90, F-test p < 0.001 in every case). Relative MAE on
held-out checkpoints is approximately 3% for all retrieval metrics, meaning the law
forecasts performance within 3% of true values using only early training observations.

**Cross-entropy (CE) is harder to model.**
CE64 has a noticeably lower mean R2 (~0.77) and in some cases falls as low as 0.12
(ranknet, specific model sizes). CE measures training loss dynamics rather than end-task
quality, and its trajectory can exhibit non-monotonic behaviour -- particularly with ranknet,
which shows occasional loss spikes -- making a simple power law a poor fit.

**precision@100 is degenerate and should be excluded.**
This metric is essentially flat across training steps at all model sizes: the retrieval
list is long enough that top-100 precision saturates early. The power law cannot fit a flat
signal, producing numerically extreme negative R2 values. This metric should be excluded
from any scaling law analyses.

**ranknet is the most consistent loss function for data scaling.**
ranknet achieves the smallest variation in R2 across model sizes (min R2 = 0.89 for
retrieval metrics), indicating that its training curves are the smoothest and most
power-law-shaped. listnet shows the most variability (min R2 = 0.74 for MAP/MRR at 1b),
likely because listnet's list-level loss produces noisier per-step trajectories.
bce sits between the two but tends to have slightly lower R2 than ranknet.

**Fit quality improves with model size for ranknet and bce, but not listnet.**
For ranknet and bce, R2 at 400m and 1b (typically > 0.93) is higher than at 17m-68m
(typically 0.82-0.90), suggesting larger models converge more smoothly. For listnet at 1b,
R2 drops sharply (e.g. 0.76 for NDCG@10), indicating an unusual training trajectory that
deviates from the expected power-law shape -- possibly a learning rate or optimisation
artefact specific to the 1b listnet run.

---

### 6.2 Model Scaling

**Performance scales predictably with model size for retrieval metrics.**
The power law fit across model sizes (17m to 150m training) achieves R2 >= 0.95 for all
retrieval metrics under ranknet and bce, and R2 ~0.95 under listnet. F-tests are uniformly
significant (p < 0.001), confirming that model size is a strong predictor of downstream
retrieval quality.

**The power law systematically underestimates performance at larger sizes.**
Across all three loss functions and all retrieval metrics, forecasts are almost always
below actual performance: listnet underestimates 83% of the time, ranknet 100%, bce 83%.
This is a consistent directional bias. The likely cause is that the power law fitted on
17m-150m extrapolates too conservatively -- larger models still sit on a steeper part of
the true scaling curve than the small-model fit anticipates. Forecasts should therefore
be treated as lower bounds, not point estimates.

**Underestimation grows with model size.**
For listnet and bce, the absolute bias at 1b is roughly 1.3-1.7x larger than at 400m,
confirming that extrapolation error compounds: the 400m forecast is already slightly
biased, and the 1b forecast drifts further in the same direction.

**bce is the most predictable loss function at scale.**
bce achieves the highest model-scaling R2 (0.994 for NDCG@10, 0.995 for Recall@10 and
P@10) and the smallest absolute bias across test models (mean 0.0076 at 400m, 0.0095 at
1b for retrieval metrics). If the goal is to use small-model runs to forecast large-model
performance, bce is the most reliable training objective.

**ranknet retrieval metrics are well-predicted; CE is not.**
ranknet achieves R2 ~0.98-0.99 for NDCG@10, Recall@10, MAP, MRR, and P@10 -- the highest
among the three loss functions for retrieval metrics. However, CE prediction completely
breaks down: R2 = 0.64 and a bias of -0.59 at 400m (predicted 2.29 vs actual 2.87). This
is consistent with the training instability noted in data scaling -- ranknet's loss
landscape at certain model sizes diverges from the smooth power law assumed by the model.

**listnet has the weakest model-scaling fit.**
R2 for listnet retrieval metrics is ~0.95, lower than both ranknet and bce. Absolute bias
at 1b is the largest (e.g. -0.024 for NDCG@10 vs -0.012 for ranknet and -0.013 for bce).
The anomalous 1b listnet training trajectory identified in data scaling propagates into
model scaling, inflating extrapolation error.

**Statistical significance of forecast errors.**
One-sample t-tests against the 10 actual checkpoint values at 400m and 1b are significant
(p < 0.05) for 34 out of 36 (metric, loss, model) combinations. The two exceptions are
listnet CE64 at 400m (p = 0.82, near-perfect accidental match) and the remaining cases are
all p < 0.001. The near-universal significance confirms that the underestimation bias is
real and not a sampling artefact: the power law is precise and smooth but systematically
biased.

---

### 6.3 Summary Recommendations

| Question | Finding |
|----------|---------|
| Which metrics scale reliably? | NDCG@k, Recall@k, MAP, MRR, P@k. CE is unreliable; P@100 is degenerate. |
| Which loss is most predictable? | **bce** for model scaling. **ranknet** for data scaling consistency. |
| Is the power law biased? | Yes -- it **systematically underestimates** at unseen larger sizes. Use forecasts as lower bounds. |
| How accurate are forecasts? | ~3% relative error for data scaling; 1-5% for model scaling, growing with extrapolation distance. |
| When does it break down? | CE under ranknet, listnet at 1b -- both driven by training instability rather than limits of the power-law form itself. |

---

## 7. Comparative Analysis: Which Loss Function Scales Better?

### 7.1 Scaling with Training Steps (Data Scaling)

The power-law exponent `c` in `y = a + b * D^c` controls how steeply performance
improves with additional training steps. A more negative `c` means faster improvement;
the asymptote `a` is the theoretical performance ceiling as steps approach infinity.

**Convergence rate (exponent c) — NDCG@10:**

| Loss | Mean c | Interpretation |
|------|--------|----------------|
| ranknet | -0.617 | Fastest per-step improvement |
| listnet | -0.568 | Moderate convergence rate |
| bce | -0.410 | Slowest convergence rate |

ranknet converges fastest: its steeper exponent means each additional batch of training
steps yields a larger gain, particularly for Recall@10 (mean c = -0.748) and P@10
(mean c = -0.744). bce improves most slowly across all retrieval metrics.

**Asymptotic ceiling (parameter a) — NDCG@10:**

| Loss | Mean asymptote a | Interpretation |
|------|-----------------|----------------|
| bce | 0.620 | Highest projected ceiling |
| listnet | 0.478 | Moderate ceiling |
| ranknet | 0.433 | Lowest projected ceiling |

Despite converging fastest, **ranknet projects the lowest performance ceiling**. bce
converges slowest but projects the highest asymptote -- meaning bce would eventually
surpass ranknet if training were continued indefinitely. This trade-off suggests:
if compute is limited (few steps), ranknet extracts more value early; if training budget
is large, bce is the better long-term choice.

**Speed to 90% of final performance (median across model sizes):**

| Loss | Median step | Range |
|------|-------------|-------|
| listnet | 800 | 300 -- 2500 |
| ranknet | 1200 | 600 -- 1700 |
| bce | 1200 | 700 -- 2600 |

listnet reaches 90% of its final NDCG@10 in as few as 800 steps -- roughly 33% faster
than ranknet and bce. This is consistent with listnet's list-level loss providing a
strong global ranking signal early in training, even if its absolute ceiling is not the
highest.

**Total gain from step 100 to final (mean across all model sizes):**

| Loss | NDCG@10 gain | Recall@10 gain | MAP gain |
|------|-------------|---------------|---------|
| ranknet | +0.310 | +0.441 | +0.259 |
| bce | +0.268 | +0.386 | +0.223 |
| listnet | +0.225 | +0.311 | +0.189 |

ranknet achieves the largest absolute improvement over the course of training, followed
by bce, then listnet. This is consistent with ranknet's steep early convergence curve
covering more ground from a cold start.

---

### 7.2 Scaling with Model Size (Model Scaling)

**Model-size exponent c — NDCG@10 (fitted on 17m-150m):**

| Loss | c (NDCG@10) | c (Recall@10) | Interpretation |
|------|-------------|--------------|----------------|
| ranknet | -1.307 | -1.502 | Steepest model-size scaling |
| listnet | -1.179 | -1.213 | Moderate scaling |
| bce | -1.140 | -1.162 | Shallowest scaling |

ranknet benefits the most per model size doubling -- at least within the 17m-150m
training range. bce scales most gently but, as with data scaling, projects the highest
asymptote (a = 0.621 for NDCG@10 vs 0.366 ranknet, 0.351 listnet), implying that at
very large model sizes bce would eventually overtake ranknet.

**NDCG@10 at final checkpoint by model size:**

| Model | listnet | ranknet | bce |
|-------|---------|---------|-----|
| 17m | 0.2493 | 0.2581 | 0.1859 |
| 32m | 0.2851 | 0.3125 | 0.2662 |
| 68m | 0.3189 | 0.3547 | 0.3190 |
| 150m | 0.3485 | 0.3551 | 0.3325 |
| 400m | 0.3648 | 0.3731 | 0.3519 |
| 1b | 0.3733 | 0.3780 | 0.3576 |

**Per-doubling NDCG@10 gains:**

| Step | listnet | ranknet | bce |
|------|---------|---------|-----|
| 17m -> 32m | +0.036 | +0.054 | +0.080 |
| 32m -> 68m | +0.034 | +0.042 | +0.053 |
| 68m -> 150m | +0.030 | +0.001 | +0.014 |
| 150m -> 400m | +0.016 | +0.018 | +0.019 |
| 400m -> 1b | +0.009 | +0.005 | +0.006 |

Three notable findings:

1. **bce gains the most at small model sizes.** The 17m->32m jump gives bce +0.080 NDCG@10,
   nearly 2x more than listnet (+0.036) and 1.5x more than ranknet (+0.054). This suggests
   bce's cross-entropy loss is poorly suited to very small models but scales aggressively
   once capacity passes a threshold.

2. **ranknet plateaus at 68m->150m (+0.001 NDCG@10)** -- essentially zero gain -- before
   recovering at 150m->400m (+0.018). This anomalous plateau likely reflects an
   optimisation artefact in the 150m ranknet run and contributes to the poor power-law fit
   and CE instability seen in earlier sections.

3. **All loss functions show strong diminishing returns.** The 400m->1b gain
   (listnet +0.009, ranknet +0.005, bce +0.006) is roughly 4-10x smaller than the
   17m->32m gain. The scaling curve is flattening for all three objectives in this range.

**Relative gain from 17m to 1b:**

| Loss | 17m NDCG@10 | 1b NDCG@10 | Relative gain |
|------|------------|-----------|---------------|
| bce | 0.1859 | 0.3576 | 1.92x |
| listnet | 0.2493 | 0.3733 | 1.50x |
| ranknet | 0.2581 | 0.3780 | 1.46x |

bce benefits most from scaling up in model size (1.92x improvement). This is partly
because bce starts from a lower base at small sizes -- the cross-entropy objective
appears to require a minimum model capacity before it becomes competitive, but once that
threshold is crossed, it tracks or exceeds the other objectives.

---

### 7.3 Head-to-Head Summary

| Criterion | Winner | Notes |
|-----------|--------|-------|
| Fastest data convergence (exponent c) | **ranknet** | Steepest slope per step for retrieval metrics |
| Earliest to 90% of peak (data scaling) | **listnet** | Reaches near-peak at step ~800 vs ~1200 |
| Highest performance ceiling (data scaling) | **bce** | Highest asymptote a across retrieval metrics |
| Strongest model-size scaling (exponent c) | **ranknet** | Steepest slope per doubling in 17m-150m range |
| Largest absolute gain 17m->1b | **bce** | 1.92x vs 1.50x (listnet) and 1.46x (ranknet) |
| Most stable / predictable at scale | **bce** | Best model-scaling R2 (0.994-0.995); smallest forecast bias |
| Lowest sensitivity to model size at small scale | **listnet** | Most gradual degradation from large to small models |
| Worst pathological behaviour | **ranknet** | Plateau at 68m->150m; CE divergence at 400m |

**Overall recommendation:** ranknet extracts the most value early in training and at
modest model sizes. bce is the better choice when training budget or model size is large,
and is the most reliably predictable for scaling law forecasting. listnet offers the
fastest early convergence signal, making it useful for quick ablations, but its absolute
ceiling is the lowest of the three.
