"""Refit the everyday priors G&T (2006) do not report, from their sources' current equivalents:

- House terms: total years in the House for members first serving from 1945 who are no longer serving
  (G&T: bioguide.congress.gov, 2,150 members since 1945), from unitedstates/congress-legislators'
  legislators-historical.yaml. Erlang (shape 2) MLE: beta = mean / 2.
- Movie run times: G&T used 233 weekly top-10 box-office movies, 1998-2003. IMDb's public datasets have
  no box-office ranks, so this takes the 40 most-voted movies of each of those years. Gaussian MLE.
- Pregnancy: US singleton births of 2022 by completed weeks of gestation (NVSR vol. 73 no. 1, 2024, Table 1:
  <34, 34-36, 37-38, 39-40, >=41 weeks). Gaussian, binned MLE (completed week w covers [w, w + 1)).
- Company lifetimes (no fit here): publicly traded North American firms have an age-independent hazard with a
  Kaplan-Meier half-life of 10.46 years (Daepp et al. 2015, J. R. Soc. Interface, Table 3), so an exponential
  lifetime of mean 10.46 / ln 2 = 15.1 years: Erlang beta = 15.1 for a firm met at a random point of its life.

    uv run python src/notebooks/fit_everyday_priors.py outputs/everyday/data
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import yaml

data = Path(sys.argv[1])

members = yaml.load(open(data / "legislators-historical.yaml"), Loader=yaml.CSafeLoader)
years = []
for m in members:
    house = [t for t in m["terms"] if t["type"] == "rep"]
    if house and date.fromisoformat(house[0]["start"]).year >= 1945:
        years.append(sum((date.fromisoformat(t["end"]) - date.fromisoformat(t["start"])).days for t in house) / 365.25)
years = np.array(years)
print(f"House: {len(years)} members, mean {years.mean():.2f} years -> Erlang beta = {years.mean() / 2:.2f}")

basics = pd.read_csv(data / "title.basics.tsv.gz", sep="\t", na_values="\\N", quoting=3,
                     usecols=["tconst", "titleType", "startYear", "runtimeMinutes"], low_memory=False)
movies = basics[(basics.titleType == "movie") & basics.startYear.between(1998, 2003)].dropna(subset=["runtimeMinutes"])
votes = pd.read_csv(data / "title.ratings.tsv.gz", sep="\t", usecols=["tconst", "numVotes"])
top = movies.merge(votes, on="tconst").sort_values("numVotes", ascending=False).groupby("startYear").head(40)
rt = top.runtimeMinutes.astype(float).to_numpy()
print(f"Run times: {len(rt)} movies, Gaussian mu = {rt.mean():.1f}, sigma = {rt.std():.1f}")

edges = np.array([-np.inf, 34, 37, 39, 41, np.inf]); pct = np.array([2.16, 6.51, 29.07, 57.11, 5.15])
def binned_nll(x):
    cdf = norm.cdf(edges, x[0], np.exp(x[1]))
    return -(pct * np.log(np.diff(cdf))).sum()
mu, log_sigma = minimize(binned_nll, [39.0, np.log(2.0)], method="Nelder-Mead").x
print(f"Pregnancy: NVSR 2022 singleton bins {pct.tolist()}, Gaussian mu = {mu:.2f}, sigma = {np.exp(log_sigma):.2f}")
print(f"Companies: Erlang beta = 10.46 / ln 2 = {10.46 / np.log(2):.1f}")
