## ✅ Train CLI Commands
Here are example command-line invocations for **training** and **testing** each of your three model types. Adjust paths, dataset names or hyperparameters as needed.

---


Below is a concise recommendation based on the imbalance ratios (IR) you’ve computed and common IR “severity” thresholds in the machine-learning literature.

## Imbalance Ratio Thresholds

Studies on imbalanced learning often categorize IRs as follows:

* **Mild skew (IR < 5):** Standard classifiers typically cope well; simple class-weighting or mild resampling may suffice ([en.wikipedia.org][1]).
* **Moderate skew (5 ≤ IR < 10):** Performance degradation on minority classes becomes noticeable; systematic oversampling (e.g. SMOTE) or cost-sensitive methods are recommended ([link.springer.com][2]).
* **Severe skew (IR ≥ 10):** Requires aggressive imbalance handling (advanced resampling, ensemble or anomaly-detection approaches) to avoid majority bias ([link.springer.com][2]).

## Dataset IR Comparison

From your table:

| Dataset                | Imbalance Ratio (IR) |
| ---------------------- | -------------------- |
| Botswana               | 3.00                 |
| Houston13              | 3.94                 |
| SalinasA               | 4.00                 |
| Kennedy\_Space\_Center | 9.20                 |
| Salinas                | 12.51                |
| Pavia\_University      | 19.83                |
| Pavia\_Centre          | 24.61                |
| Indian\_Pines          | 24.40                |


* **Below mild threshold (IR < 5):** Botswana (3.00), SalinasA (4.00)
* **Moderate skew (5 ≤ IR < 10):** Kennedy\_Space\_Center (9.20)
* **Severe skew (IR ≥ 10):** Salinas, Pavia datasets, Indian\_Pines

## Recommendation

For an initial experiment with minimal imbalance handling overhead, **choose the Botswana dataset** (IR = 3.00). It sits comfortably in the “mild skew” regime, yet still offers realistic hyperspectral challenges and is a common benchmark in HSI classification studies ([mdpi.com][3]).

If you’d like a slightly more challenging but still manageable case, you may also consider **SalinasA** (IR = 4.00), which remains in the mild‐skew category.

All other datasets have IRs ≥ 5 and will demand more involved imbalance mitigation strategies (e.g., SMOTE, cost-sensitive losses, ensemble sampling) before training a robust classifier.

[1]: https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis "Oversampling and undersampling in data analysis"
[2]: https://link.springer.com/article/10.1007/s13748-016-0094-0 "Learning from imbalanced data: open challenges and future directions"
[3]: https://www.mdpi.com/2072-4292/14/24/6406 "A Hypered Deep-Learning-Based Model of Hyperspectral Images ..."
