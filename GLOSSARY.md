# Fraud API

Scores bank account opening applications for fraud risk, using a model trained on the public Bank Account Fraud (BAF) dataset.

## Applications

**Application**:
A single request to open a bank account, described by what the Applicant submitted and by signals about the device and session it came from.
_Avoid_: transaction, account, customer, request

**Applicant**:
The person who submits an Application. An Applicant is not yet a customer of the bank.
_Avoid_: customer, client, user, account holder

**Fraudulent Application**:
An Application later confirmed to be fraud. This is the label the model learns to predict.
_Avoid_: fraud transaction, scam, bad application

**Legitimate Application**:
An Application that was not confirmed as fraud.
_Avoid_: honest application, good application, normal application

## Scoring

**Fraud Score**:
A number from 0 to 1 that ranks Applications by fraud risk: higher means riskier. It orders Applications; it is not a probability.
_Avoid_: probability, fraud probability, risk percentage

**Threshold**:
The Fraud Score at or above which an Application is sent to Review. It is chosen so that 5% of Legitimate Applications in the Validation Month would be sent to Review.
_Avoid_: cutoff, decision boundary

**Decision**:
What the service concludes about an Application: Pass or Review.
_Avoid_: verdict, prediction, outcome

**Pass**:
The Decision that the Application shows no sign of fraud worth checking. It is not an approval of the account.
_Avoid_: approve, accept, allow

**Review**:
The Decision to send an Application to manual checking because its Fraud Score reached the Threshold. An Application with this Decision is said to be flagged.
_Avoid_: decline, reject, block, deny

**Top Factor**:
One of the few fields that moved an Application's Fraud Score the most, with how far it moved the model's raw score (its contribution, in log-odds). It explains the model's ranking, not what causes fraud.
_Avoid_: reason, cause, probability contribution

**Model Version**:
A trained model together with its Threshold, identified so that every Decision can be traced to the model that made it.
_Avoid_: release, build

## Time

**Month**:
The relative month (0 to 7) in which an Application was submitted. It is the only notion of time in the data.
_Avoid_: period, date, timestamp

**Training Months**:
The Months whose Applications the final model learns from.

**Tuning Month**:
The last Training Month, held out while comparing model configurations so that configuration choices never touch the Validation Month.
_Avoid_: inner validation, CV fold

**Validation Month**:
The Month held out from training and used only to choose the Threshold.
_Avoid_: dev set, holdout

**Test Months**:
The latest Months, held out from training, tuning and Threshold selection, used once to report how the model performs on future Applications.
_Avoid_: holdout, out-of-time set

## Evaluation

**Recall@5%FPR**:
The share of Fraudulent Applications flagged when at most 5% of Legitimate Applications are flagged. The primary measure of model quality.
_Avoid_: accuracy, detection rate

**Age Group**:
One of two groups of Applicants compared for fairness: aged 50 or over, and under 50.
_Avoid_: seniors, elderly, protected group

**FPR Ratio**:
The false positive rate of the Age Group with the lower rate divided by that of the Age Group with the higher rate. 1 means Legitimate Applications from both groups are flagged equally often; 0.3 means one group's are flagged about three times as often.
_Avoid_: FPR disparity, bias, fairness score, predictive equality

## Serving

**Prediction Log**:
The durable record of every Application the service scored and what it answered.
_Avoid_: audit log, history, journal
