# Keep age as a feature and report the FPR Ratio

`customer_age` stays a model feature, and every evaluation reports the FPR Ratio between Age Groups next to Recall@5%FPR. Applicants aged 50 or over are almost three times as likely to submit a Fraudulent Application (2.3% against 0.8%), so the model is expected to flag their Legitimate Applications more often; the BAF authors report an FPR Ratio around 0.3 for their best models. We choose to measure and publish that disparity rather than hide it.

## Considered Options

- **Drop age from the features ("fairness through unawareness").** Usually leaves most of the disparity in place, because other features act as proxies for age, while costing recall. Comparing the model with and without age is a planned follow-up experiment, not the default.
- **A separate Threshold per Age Group, to equalise FPR.** Treating Applicants differently because of their age is legally contentious in financial services, so it is ruled out.
