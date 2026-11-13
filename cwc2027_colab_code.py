# %% [markdown]
# # ICC Men's Cricket World Cup 2027 — Colab EDA & Monte Carlo Simulator
# Hosts: South Africa, Zimbabwe & Namibia (Oct 2 – Nov 21, 2027)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sched = pd.read_csv("cwc_2027_schedule_venues.csv")
teams = pd.read_csv("cwc_2027_team_ratings.csv")
print("Schedule rows:", len(sched), "| Teams:", len(teams))
sched.head()

# %%
# Venue capacity & host country breakdown
venue_summary = sched.groupby(["venue", "host_country", "stadium_capacity"]).size().reset_index(name="fixtures")
plt.figure(figsize=(10, 5))
sns.barplot(data=venue_summary.sort_values("stadium_capacity", ascending=False), x="stadium_capacity", y="venue", hue="host_country")
plt.title("2027 Cricket World Cup — 12 Host Venues by Capacity")
plt.show()

# %%
# Monte Carlo Simulation (1,000 tournaments)
rating_map = dict(zip(teams["team"], teams["odi_rating"]))

def win_prob(t1, t2, host_country, host_boost=6.0):
    r1, r2 = rating_map.get(t1, 60), rating_map.get(t2, 60)
    if (t1 == "South Africa" and host_country == "South Africa") or (t1 == "Zimbabwe" and host_country == "Zimbabwe"):
        r1 += host_boost
    if (t2 == "South Africa" and host_country == "South Africa") or (t2 == "Zimbabwe" and host_country == "Zimbabwe"):
        r2 += host_boost
    return 1.0 / (1.0 + np.exp(-0.065 * (r1 - r2)))

rng = np.random.default_rng(42)
group_a = teams[teams["group_2027"] == "Group A"]["team"].tolist()
group_b = teams[teams["group_2027"] == "Group B"]["team"].tolist()
all_t = group_a + group_b
champ_counts = {t: 0 for t in all_t}
gs = sched[sched["stage"] == "Group Stage"]

for _ in range(1000):
    pts = {t: 0.0 for t in all_t}
    for _, m in gs.iterrows():
        if rng.random() < win_prob(m["team_1"], m["team_2"], m["host_country"]):
            pts[m["team_1"]] += 2.0 + rng.uniform(0, 0.05)
        else:
            pts[m["team_2"]] += 2.0 + rng.uniform(0, 0.05)
    ga = sorted(group_a, key=lambda x: pts[x], reverse=True)
    gb = sorted(group_b, key=lambda x: pts[x], reverse=True)
    best4 = ga[3] if pts[ga[3]] >= pts[gb[3]] else gb[3]
    s7 = ga[:3] + gb[:3] + [best4]
    s7_pts = {t: 0.0 for t in s7}
    for i in range(len(s7)):
        for j in range(i + 1, len(s7)):
            if rng.random() < win_prob(s7[i], s7[j], "South Africa"):
                s7_pts[s7[i]] += 2.0 + rng.uniform(0, 0.05)
            else:
                s7_pts[s7[j]] += 2.0 + rng.uniform(0, 0.05)
    sf = sorted(s7, key=lambda x: s7_pts[x], reverse=True)[:4]
    w1 = sf[0] if rng.random() < win_prob(sf[0], sf[3], "South Africa") else sf[3]
    w2 = sf[1] if rng.random() < win_prob(sf[1], sf[2], "South Africa") else sf[2]
    champ = w1 if rng.random() < win_prob(w1, w2, "South Africa") else w2
    champ_counts[champ] += 1

res = pd.DataFrame([{"Team": k, "Title_Prob_%": round(v / 10.0, 1)} for k, v in champ_counts.items()]).sort_values("Title_Prob_%", ascending=False)
print(res)