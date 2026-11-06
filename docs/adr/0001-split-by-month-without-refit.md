# Split by Month, and never refit after choosing the Threshold

The final model trains on Months 0–4, the Threshold is chosen on Month 5 (the Validation Month) so that FPR there is 5%, and Months 6–7 (the Test Months) are scored once for the final report. Hyperparameters are searched with Optuna on an inner split (train on Months 0–3, compare on Month 4, the Tuning Month), so neither the Validation Month nor the Test Months influence configuration choices.

We deliberately do not retrain on Months 0–5 once the Threshold is chosen, although that would add a month of data: retraining shifts the Fraud Score distribution, so the chosen Threshold would no longer correspond to 5% FPR, and nothing would be left to choose a new one on.

## Considered Options

- **Train 0–5, test 6–7 (the BAF paper's protocol), Threshold by cross-validation inside 0–5.** Random folds mix Months, so the Threshold would be chosen on data from the same period it was trained on and would not reflect drift.
- **Choose the Threshold, then refit on 0–5.** Rejected for the reason above.
