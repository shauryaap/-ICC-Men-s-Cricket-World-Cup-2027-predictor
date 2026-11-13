# ICC Men's Cricket World Cup 2027 — Schedule, Venue & Monte Carlo Predictor

Business Analytics Group Project built with **Google Colab**, hosted on **GitHub**, and deployed on **Streamlit Community Cloud**.

## Verified Data Sources
- **2027 Schedule, Format, 12 Venues & Qualified Teams:** [Wikipedia — 2027 Cricket World Cup](https://en.wikipedia.org/wiki/2027_Cricket_World_Cup) and [ICC Official Fixture Announcement](https://www.icc-cricket.com/tournaments/icc-cricket-world-cup-2027/news/fixture-schedule-announced-for-icc-men-s-cricket-world-cup-2027)
- **2026 ICC Men's ODI Team Rankings:** [ICC Men's ODI Team Rankings](https://www.icc-cricket.com/rankings) and [myKhel ICC ODI Rankings](https://telugu.mykhel.com/cricket/icc-odi-rankings/)

## Files to Upload to GitHub
Rename `app_cwc2027.py` to `app.py` and `cwc2027_requirements.txt` to `requirements.txt` (or set `app_cwc2027.py` as your Main file path in Streamlit Cloud), alongside:
- `cwc_2027_schedule_venues.csv`
- `cwc_2027_team_ratings.csv`
- `cwc2027_colab_code.py` (or your `.ipynb` from Google Colab)

## 4-Member Viva Split
1. **Member 1 (Data & Elo Probability Engine):** Explains `win_probability()` logistic formula, ICC ODI ratings, and home-ground boost for South Africa and Zimbabwe.
2. **Member 2 (Schedule & 12-Venue EDA — Tab 1):** Explains the 3-country host split (8 SA, 3 ZIM, 1 NAM), stadium capacities, and pitch profiles.
3. **Member 3 (Fixture Predictor & Monte Carlo Simulator — Tabs 2 & 3):** Explains how 1,000+ simulated tournaments model Group Stage → Super 7 → Semi-Finals → Final probabilities.
4. **Member 4 (K-Means Venue Clustering & Deployment — Tab 4):** Explains `StandardScaler`, `KMeans(n_clusters=3)`, `@st.cache_data`, and Streamlit Cloud deployment.
