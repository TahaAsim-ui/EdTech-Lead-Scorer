# ExtraaLearn Lead Conversion Scorer

A machine learning web app that predicts which leads are most likely to convert to paid customers for ExtraaLearn, an EdTech startup offering upskilling programs in cutting-edge technologies.

## Overview

ExtraaLearn generates a large number of leads regularly but needed a way to identify which leads were worth prioritising. This app uses a tuned Random Forest classifier to score each lead with a conversion probability and assign a priority tier, so the sales team knows exactly where to focus their efforts.

## Features

- **Score Leads** — Upload a CSV of leads and instantly get a ranked table with conversion probabilities and priority tiers (High / Medium / Low)
- **Data Insights** — Visual breakdowns of your uploaded leads: occupation mix, first interaction channel, profile completion, High vs Low priority comparisons, and average conversion probability by segment
- **Model Insights** — Feature importance chart showing which variables drive conversion, available at any time without uploading data
- **Download Results** — Export the scored and ranked leads as a CSV

## The Model

The app uses a Random Forest classifier trained on 4,612 historical leads with the following key findings:

| Feature | Importance |
|---|---|
| Time spent on website | ~30% |
| First interaction via Website | ~28% |
| Profile completion (Medium) | ~19% |
| Age | ~6% |
| Other features | ~17% combined |

**Model performance (test set):**
- Accuracy: 85%
- Recall for converted leads: 88%
- F1 score for converted leads: 0.77

Recall was prioritised as the key metric since missing a real potential customer is a greater business loss than a false positive.

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/TahaAsim-ui/EdTech-Lead-Scorer.git
cd EdTech-Lead-Scorer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Input Data Format

Upload a CSV with the following columns:

| Column | Type | Description |
|---|---|---|
| age | Integer | Age of the lead |
| current_occupation | String | Professional / Student / Unemployed |
| first_interaction | String | Website / Mobile App |
| profile_completed | String | High / Medium / Low |
| website_visits | Integer | Number of website visits |
| time_spent_on_website | Integer | Total time on site (seconds) |
| page_views_per_visit | Float | Average pages viewed per visit |
| last_activity | String | Email Activity / Phone Activity / Website Activity |
| print_media_type1 | String | Yes / No |
| print_media_type2 | String | Yes / No |
| digital_media | String | Yes / No |
| educational_channels | String | Yes / No |
| referral | String | Yes / No |

An optional `ID` column is preserved in the output if present. An optional `status` column is ignored if present.

## Priority Tiers

| Tier | Conversion Probability | Action |
|---|---|---|
| 🔴 High | ≥ 60% | Prioritise for direct sales outreach |
| 🟡 Medium | 40–59% | Targeted email nudge |
| 🟢 Low | < 40% | Long-term nurture sequence |

## Tech Stack

- **Python** — pandas, scikit-learn, numpy
- **Streamlit** — web app framework
- **Plotly** — interactive visualisations
- **Random Forest** — classification model (scikit-learn)
