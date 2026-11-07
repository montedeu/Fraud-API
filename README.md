# Fraud API

A public API that scores bank account opening Applications for fraud risk. Send one Application, get back a Fraud Score, a Decision (`pass` or `review`) and the Model Version that made it. Behind it is a LightGBM model trained on the public [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) dataset, served by FastAPI on Google Cloud Run. Every scored Application is written to a Prediction Log in Postgres.

<!-- generated:about -->
Model Version `lgbm-f5421b309a8c`, Threshold 0.1037.
<!-- /generated:about -->

**Try it:** open the live [`/docs`](https://fraud-api-r6bn3ptcuq-ew.a.run.app/docs), pick `POST /predict` → *Try it out* → *Execute*. Or from a terminal:

<!-- generated:curl -->
```bash
curl -s https://fraud-api-r6bn3ptcuq-ew.a.run.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"income": 0.6, "name_email_similarity": 0.5, "prev_address_months_count": 20, "current_address_months_count": 60, "customer_age": 30, "days_since_request": 0.02, "intended_balcon_amount": 10.5, "payment_type": "AA", "zip_count_4w": 1500, "velocity_6h": 5000.0, "velocity_24h": 4800.0, "velocity_4w": 5000.0, "bank_branch_count_8w": 10, "date_of_birth_distinct_emails_4w": 8, "employment_status": "CA", "credit_risk_score": 120, "email_is_free": 1, "housing_status": "BA", "phone_home_valid": 0, "phone_mobile_valid": 1, "bank_months_count": 10, "has_other_cards": 0, "proposed_credit_limit": 500.0, "foreign_request": 0, "source": "INTERNET", "session_length_in_minutes": 5.0, "device_os": "linux", "keep_alive_session": 1, "device_distinct_emails_8w": 1}'
```

```json
{"fraud_score": 0.008908066429150555, "decision": "pass", "model_version": "lgbm-f5421b309a8c", "request_id": "…"}
```
<!-- /generated:curl -->

A Fraud Score ranks Applications; it is not a probability. An Application is sent to `review` when its Fraud Score reaches the Threshold, chosen so that 5% of Legitimate Applications in the Validation Month would be. Send `null` for anything the Applicant did not provide. Unknown categories, out-of-range values, and missing or extra fields get a 422 that names the field.

The words used here (Application, Fraud Score, Threshold, Decision, Month…) are defined in [GLOSSARY.md](GLOSSARY.md).

## How the model was built

The data has eight Months (0–7) and nothing finer, so the split is by Month ([ADR 0001](docs/adr/0001-split-by-month-without-refit.md)):

| Months | Role |
|---|---|
| 0–4 | Training Months. The final model is fitted here. |
| 4 | Tuning Month. Optuna compares configurations trained on Months 0–3 here. |
| 5 | Validation Month. Used only to choose the Threshold. |
| 6–7 | Test Months. Scored once, after every choice was fixed. |

The model is not refitted on Month 5 after the Threshold is chosen, because that would move the Fraud Scores away from the Threshold chosen for them. Two baselines get their own Thresholds by the same rule: a logistic regression (median imputation with missing indicators, scaling, one-hot) and a single-feature rule on `credit_risk_score`.

The trained model and its metadata are committed in [`model/`](model) ([ADR 0002](docs/adr/0002-model-artifact-committed-to-repo.md)), so one commit pins the code and the exact model it serves. Every number below is generated from [`model/metadata.json`](model/metadata.json) by `fraud-api readme`, and a test fails if the README drifts from it.

## Results

<!-- generated:test-results -->
Test Months 6–7: 205,011 Applications, 2,878 of them Fraudulent (1.4%). Each model uses its own Threshold, chosen on the Validation Month.

| Model | Recall@5%FPR | Recall at the Threshold | FPR at the Threshold | PR-AUC | FPR Ratio |
|---|---:|---:|---:|---:|---:|
| LightGBM (served) | 55.0% | 59.1% | 6.0% | 0.198 | 0.32 |
| Logistic regression | 50.9% | 50.0% | 4.8% | 0.163 | 0.29 |
| `credit_risk_score` rule | 19.8% | 19.8% | 4.9% | 0.037 | 0.48 |
<!-- /generated:test-results -->

- **Recall@5%FPR** is the share of Fraudulent Applications flagged at the Fraud Score that flags 5% of Legitimate Applications in these Months. It measures ranking quality.
- **Recall and FPR at the Threshold** are what the served model actually does with the Threshold chosen a Month earlier.
- **FPR Ratio** compares the two Age Groups (50 or over, under 50) at the Threshold: the lower group's false positive rate divided by the higher one's. 1 is parity. The BAF authors report about **0.3** for their best LightGBM models on this data.

<!-- generated:validation-results -->
Validation Month 5: 119,323 Applications, 1,411 of them Fraudulent (1.2%). Optimistic: each Threshold was chosen on this Month, so FPR at the Threshold is 5% by construction.

| Model | Recall@5%FPR | Recall at the Threshold | FPR at the Threshold | PR-AUC | FPR Ratio |
|---|---:|---:|---:|---:|---:|
| LightGBM (served) | 54.8% | 54.8% | 5.0% | 0.193 | 0.33 |
| Logistic regression | 50.7% | 50.7% | 5.0% | 0.158 | 0.32 |
| `credit_risk_score` rule | 19.3% | 19.3% | 4.9% | 0.031 | 0.46 |
<!-- /generated:validation-results -->

### Tuning

<!-- generated:tuning -->
50 TPE trials, each trained on Months 0–3 and scored on the Tuning Month 4. Recall@5%FPR on the Tuning Month went from 54.3% with LightGBM defaults to 59.1% tuned (+4.8 points). The tuned figure is the best of 50 trials on that Month, so part of the gain is selection; the Validation Month and Test Months above are the honest measure. The final model uses the best trial's 646 trees. Most of the variance between trials came from `learning_rate` 81%, `num_leaves` 9%, `scale_pos_weight` 6%.
<!-- /generated:tuning -->

![Hyperparameter importance from the Optuna study](reports/param_importance.png)

All trials are in [`reports/optuna_trials.csv`](reports/optuna_trials.csv).

### What the numbers say

<!-- generated:discussion -->
- **LightGBM against logistic regression.** On the Test Months LightGBM flags 55.0% of Fraudulent Applications at 5% FPR, logistic regression 50.9%: 4.2 points. PR-AUC is 0.198 against 0.163. The gap is real but modest: a well-preprocessed linear model gets most of the way, and the single-feature rule gets 19.8%.
- **Age disparity.** At its Threshold, LightGBM's FPR Ratio on the Test Months is 0.32: Legitimate Applications from one Age Group are flagged about 3.1 times as often as from the other (fraud rises with age, and the model learns it: EDA section 6). That is in line with the BAF authors' best models, around 0.3, and is published rather than hidden ([ADR 0003](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md)).
- **Drift between Months.** The share of Fraudulent Applications rises from 1.0% in the Training Months and 1.2% in the Validation Month to 1.4% in the Test Months. The Threshold chosen for 5% FPR on the Validation Month flags 6.0% of Legitimate Applications on the Test Months, and catches 59.1% of Fraudulent ones: the operating point moves, and a real deployment would re-choose the Threshold on recent data.
<!-- /generated:discussion -->

Age stays a feature on purpose ([ADR 0003](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md)). Several other fields carry part of the age signal (EDA section 7), so dropping it is expected to leave much of the disparity in place while costing recall; measuring that is a planned follow-up. A separate Threshold per Age Group would treat Applicants differently because of their age.

The [EDA notebook](notebooks/eda.ipynb) shows the data behind these choices; each section ends with the decision it led to.

## Data caveats

- **Synthetic data.** BAF is a synthetic dataset generated from real bank account opening data, with privacy noise added. Its [datasheet](https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf) says it is meant for research and benchmarking, not for building real-world fraud models. This project is a demonstration of the engineering around a model, not a fraud model to deploy.
- **Missing values.** The dataset encodes missing values as -1 in five columns (`prev_address_months_count`, `current_address_months_count`, `bank_months_count`, `session_length_in_minutes`, `device_distinct_emails_8w`). Negative `intended_balcon_amount` values are the same sentinel blurred by the privacy noise. All of these become real missing values before the model sees them, and the API takes `null` for them, so clients never send dataset-specific sentinel numbers. Missingness itself carries signal, and LightGBM uses it directly.
- **No data in the repository.** `Base.csv` is not committed, because its redistribution terms are not stated. The reference Applications and test data are synthetic.

## Threat model

The API is public and unauthenticated by design, so a reviewer can try it in a minute. These are the risks that follow from that, and what bounds them.

- **Abuse and cost.** Cloud Run is capped at one instance with 1 vCPU and 512 MiB. Even busy around the clock, that is about 2.6 million vCPU-seconds a month: roughly **$45 a month** in compute at [Cloud Run's instance-based prices](https://cloud.google.com/run/pricing) (Tier 1, after the free tier). Outbound traffic is not capped by the instance limit. It is about $0.12 per GiB, and the largest public response, `/openapi.json`, is about 6 KB, so a million requests for it cost under a dollar. A sustained flood could still cost more than the compute cap. Two things answer that:
  - a GCP budget emails the owner at 50% and 100% of a small monthly amount;
  - the kill switch below takes the service off the internet in seconds. There is no automatic billing shutdown.
- **Rate limit.** `/predict` allows 60 requests a minute per client, with a 429 and `Retry-After` past that. Request bodies over 4 KB get a 413 before they are parsed. The client is the last `X-Forwarded-For` hop, the address Cloud Run saw. Earlier hops are client-supplied and ignored, so a forged header doesn't escape the limit. The limit is per address, so it slows one client but not a distributed attack; an IPv6 client can also rotate addresses within its /64. The limit is kept in memory, which is enough for a single instance.
- **Model extraction does not matter here.** The model file is already public in this repository, and it was trained on public synthetic data. Querying the API reveals nothing that `git clone` doesn't.
- **Secrets.**
  - The Postgres connection string lives only in Secret Manager. Only the service's runtime identity can read it, and the deploy workflow never handles it.
  - GitHub Actions deploys through Workload Identity Federation, pinned to this repository, with short-lived tokens; no service-account key exists.
  - GitHub push protection blocks committed secrets.
- **Personal data.** The Prediction Log stores the Application, the Fraud Score, the Decision and the latency. It stores no IP addresses or headers.

## Operations

The service runs as `fraud-api` in project `fraud-api-g2z9q`, region `europe-west1`. Every merge to `main` runs the checks, then the deploy:

1. builds and pushes the image;
2. deploys it as a new revision with no traffic;
3. smoke-tests that revision ([`scripts/smoke_test.py`](scripts/smoke_test.py));
4. only then moves all traffic to it.

A failed smoke test leaves callers on the previous revision. *Actions → Deploy → Run workflow* redeploys `main` without a commit.

**Kill switch:** make the service internal-only. It stops answering the internet at once:

```bash
gcloud run services update fraud-api --project fraud-api-g2z9q --region europe-west1 --ingress internal
```

To reopen it, run the same command with `--ingress all`.

**Rollback:** send all traffic to the revision before the newest one:

```bash
gcloud run services update-traffic fraud-api --project fraud-api-g2z9q --region europe-west1 --to-revisions "$(gcloud run revisions list --service fraud-api --project fraud-api-g2z9q --region europe-west1 --sort-by=~metadata.creationTimestamp --format='value(metadata.name)' | sed -n 2p)=100"
```

To go forward again, pass `--to-tags candidate=100`; the last deployed revision carries the `candidate` tag. Artifact Registry keeps the three most recent images, so the previous revisions can still start.

## Run it locally

You need [uv](https://docs.astral.sh/uv/) and Python 3.12; Docker is optional.

```bash
uv sync                                    # dependencies, including dev and train
uv run pytest                              # the tests; an embedded Postgres starts if TEST_DATABASE_URL is unset
uv run fraud-api serve                     # the committed Model Version on http://localhost:8000/docs
```

To retrain, download `Base.csv` and run the pipeline. Download needs a [Kaggle API token](https://www.kaggle.com/docs/api), and the file's SHA-256 is checked. The pipeline runs tuning, the final fit, the Threshold, the baselines and the one evaluation on the Test Months:

```bash
uv run --group kaggle fraud-api download   # into data/
uv run fraud-api train                     # rewrites model/ and reports/
uv run fraud-api readme                    # regenerates the numbers in this README
```

The same seed reproduces the same Model Version. Without `DATABASE_URL` the Prediction Log is off; with it, the table is created on first start.

In a container:

```bash
docker build -t fraud-api .
docker run --rm -p 8080:8080 -e PORT=8080 fraud-api
```

## More

- [GLOSSARY.md](GLOSSARY.md): the domain vocabulary.
- Architecture decisions:
  - [0001 Split by Month, no refit after the Threshold](docs/adr/0001-split-by-month-without-refit.md)
  - [0002 Commit the trained model](docs/adr/0002-model-artifact-committed-to-repo.md)
  - [0003 Keep age as a feature, report the FPR Ratio](docs/adr/0003-keep-age-as-feature-and-report-fpr-ratio.md)
- [EDA notebook](notebooks/eda.ipynb)
- CI: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (PRs: ruff, pytest against Postgres, image build). Deploy: [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).
- Cloud setup, reproducible: [`scripts/provision.sh`](scripts/provision.sh).
