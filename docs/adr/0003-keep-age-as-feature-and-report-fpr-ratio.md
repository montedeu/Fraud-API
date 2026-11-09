# Keep age as a feature and report the FPR Ratio and the Precision Ratio

`customer_age` stays a model feature, and every evaluation reports the FPR Ratio and the Precision Ratio between Age Groups next to Recall@5%FPR. Applicants aged 50 or over are almost three times as likely to submit a Fraudulent Application (2.21% against 0.75% in Months 0–5, EDA section 6), so the model is expected to flag their Legitimate Applications more often; the BAF authors report an FPR Ratio around 0.3 for their best models. We choose to measure and publish that disparity rather than hide it.

## Considered Options

- **Drop age from the features ("fairness through unawareness").** Usually leaves most of the disparity in place, because other features act as proxies for age, while costing recall. This was later measured; see below.
- **A separate Threshold per Age Group, to equalise FPR.** Treating Applicants differently because of their age is legally contentious in financial services, so it is ruled out.

## Measured

On Model Version `lgbm-f5421b309a8c` (8 November 2026), the training pipeline fitted the same LightGBM without `customer_age` and reported it next to the served model (README, "What the numbers say"; `experiments.without_age` in `model/metadata.json`). That run contradicts half of the reasoning above. The gap between Age Groups narrowed but stayed wide, as expected. The recall cost, however, was a fraction of a point on the Test Months, within the noise of a single fit, rather than the clear loss this ADR assumed. Revisited by the developer the same day, with those numbers: age stays a feature, and the FPR Ratio stays reported.

## The test that decides it

The 8 November revisit kept age with no reason attached. On 10 November 2026 the developer set the goal age has to serve: a Review should carry the same fraud risk whatever the Applicant's age. That is the Precision Ratio, and the rule is to keep age if it brings the Precision Ratio closer to 1 than the model without it, and to drop age otherwise.

Measured on the same Model Version (README, the paragraph after "What the numbers say"; `precision_ratio` in `model/metadata.json`): with age the Precision Ratio is higher than without it, and the paired bootstrap interval of the difference excludes zero. Age stays a feature.

The cost is the FPR Ratio, which is lower with age: Legitimate Applicants in one Age Group are sent to Review more often. Fraud is more common among Applicants aged 50 or over, so no model short of a perfect one can make both ratios 1 together with equal recall; this project chooses an equal meaning of a Review over equal flag rates for Legitimate Applicants, and publishes both ratios.
