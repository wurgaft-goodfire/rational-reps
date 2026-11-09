"""Everyday predictions (Griffiths & Tenenbaum 2006) asked as *remaining* time. Having seen t, predict
R = T - t. The posterior is p(R | t) ∝ p(t + R) / (t + R), with no support cut-off. The prior is
evaluated at t + R, so it enters the t-R interaction, and the prior family fixes the predicted code of
t up to scale: Gaussian, a line in t; power law, a function of log t (it depends on R / t); Erlang, a
single point (the prior's factor T cancels the 1/T likelihood, leaving one exponential for every t).

Each domain's prior parameters only set the scale of the code and the behavioral reference medians, not
its shape. Those marked G&T are the paper's fits; the rest are refit by fit_everyday_priors.py from the
current equivalents of G&T's sources. Poem lengths are left out: G&T's source no longer exists."""
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

_INTRO = ("Each question below gives one piece of information and asks you to predict "
                       "something from it. Give your best guess as a single number.\n")

# Worked examples before every question, asked the same way as it (how much more, or how much in total) and
# answered accordingly. They come from held-out domains with mixed prediction rules (a fixed length, Gaussian-
# like, roughly memoryless, power-law-like), so they teach the answer format and what each question asks for
# without favouring one rule. Answers are never the number in the question: zero-shot, Llama-3.1-8B puts about
# 10% of its answer mass on t itself.
# (situation, how-much-more question, answer, in-total question, answer)
SHOTS = [("A soccer match has been going for 30 minutes.", "How many more minutes will it last?", 60,
          "How many minutes will it last in total?", 90),
         ("A cake has been baking for 20 minutes.", "How many more minutes until it is done?", 25,
          "How many minutes will it bake in total?", 45),
         ("You have been waiting at a bus stop for 4 minutes.", "How many more minutes until the bus comes?", 7,
          "How many minutes will you wait in total?", 11),
         ("A TV series has aired 3 seasons so far.", "How many more seasons will it have?", 2,
          "How many seasons will it have in total?", 5),
         ("A sitcom episode has been on for 9 minutes.", "How many more minutes will it run?", 13,
          "How many minutes will it run in total?", 22)]
PREFIX = {"remaining": _INTRO + "".join(f"{s} {q} Answer: {a}\n" for s, q, a, _, _ in SHOTS),
          "total": _INTRO + "".join(f"{s} {q} Answer: {a}\n" for s, _, _, q, a in SHOTS)}

# Templates hold no digits besides {t}, and end on the space before the answer. "prior" asks for a total
# with no t, to elicit the model's own prior.
DOMAINS = {
    "lifespan": {
        "family": "gaussian", "prior": {"mu": 75, "sigma": 16},                       # G&T
        "t": range(10, 96), "R_max": 110,
        "templates": {
            "remaining": ["Insurance actuaries estimate how long people will live. A man is {t} years old. "
                          "How many more years will he live? Answer in years: ",
                          "You meet a man who is {t} years old. How many more years do you expect him to live? "
                          "Years left: ",
                          "A doctor sees a patient, a man aged {t}. How many years does he have left to live? "
                          "Answer in years: ",
                          "My grandfather's friend just turned {t}. For how many more years will he be alive? "
                          "Years left: ",
                          "A man celebrates his birthday today; he is now {t}. How many more birthdays will he "
                          "have? Number of years: "],
            "total": ["Insurance actuaries estimate how long people will live. A man is {t} years old. "
                      "At what age will he die? Answer in years: ",
                      "You meet a man who is {t} years old. How old do you expect him to be when he dies? "
                      "Age at death: ",
                      "A doctor sees a patient, a man aged {t}. How old will he be when he dies? "
                      "Answer in years: ",
                      "My grandfather's friend just turned {t}. To what age will he live? Age at death: ",
                      "A man celebrates his birthday today; he is now {t}. How old will he be at his last "
                      "birthday? Age: "],
            "prior": ["Insurance actuaries look at the records of men who have died. At what age did one of "
                      "these men die? Answer in years: ",
                      "Think of a man who has died. How old was he when he died? Age at death: ",
                      "A doctor reads the file of a male patient who has died. How old was he when he died? "
                      "Answer in years: ",
                      "My grandfather's friend passed away. To what age did he live? Age at death: ",
                      "A man has just had his last birthday before he died. How old was he? Age: "]}},
    "runtime": {
        "family": "gaussian", "prior": {"mu": 116.8, "sigma": 23.2},                  # refit, fit_everyday_priors.py
        "t": range(10, 152, 2), "R_max": 200,
        "templates": {
            "remaining": ["You drop in on a friend who has been watching a movie for {t} minutes. "
                          "How many more minutes will the movie run? Answer in minutes: ",
                          "A movie has been playing for {t} minutes. How many minutes are left until it ends? "
                          "Minutes left: ",
                          "You walk into a cinema late; the film started {t} minutes ago. How many more minutes "
                          "until the credits roll? Answer in minutes: ",
                          "We are {t} minutes into a movie on TV. How much longer will the movie go on? "
                          "Minutes left: ",
                          "A film has been running for {t} minutes. How many minutes of it remain? "
                          "Remaining minutes: "],
            "total": ["You drop in on a friend who has been watching a movie for {t} minutes. "
                      "How long is the whole movie? Answer in minutes: ",
                      "A movie has been playing for {t} minutes. How long is the movie in total? "
                      "Total minutes: ",
                      "You walk into a cinema late; the film started {t} minutes ago. How long is the film "
                      "from start to credits? Answer in minutes: ",
                      "We are {t} minutes into a movie on TV. How long is the whole movie? Total minutes: ",
                      "A film has been running for {t} minutes. What is its full running time? "
                      "Running time in minutes: "],
            "prior": ["Think of a movie that was shown in theaters. How long is the whole movie? "
                      "Answer in minutes: ",
                      "A friend picks a movie to watch. How many minutes long is the movie in total? "
                      "Total minutes: ",
                      "You buy a ticket for a film at the cinema. How long is the film from start to "
                      "credits? Answer in minutes: ",
                      "A movie comes on TV. How long is the whole movie? Total minutes: ",
                      "Pick a film released in theaters. What is its full running time? "
                      "Running time in minutes: "]}},
    "grosses": {
        "family": "power", "prior": {"gamma": 1.71},                                   # G&T (the 1.5x rule)
        "t": range(1, 101), "R_max": 999,
        "templates": {
            "remaining": ["A movie has made {t} million dollars at the box office so far. How many more "
                          "million dollars will it make? Answer in millions of dollars: ",
                          "So far a film has grossed {t} million dollars. How much more will it gross before "
                          "it leaves theaters? Additional millions: ",
                          "A newspaper reports that a new film has taken {t} million dollars in ticket sales. "
                          "How many more million dollars in ticket sales will it take? Answer in millions of "
                          "dollars: ",
                          "A studio's movie is still in theaters and has earned {t} million dollars. How much "
                          "more will it earn at the box office? Additional millions: ",
                          "A film is {t} million dollars into its theatrical run. How many million dollars more "
                          "will it bring in? Further millions: "],
            "total": ["A movie has made {t} million dollars at the box office so far. How many million "
                      "dollars will it make in total? Answer in millions of dollars: ",
                      "So far a film has grossed {t} million dollars. What will its total gross be when it "
                      "leaves theaters? Total millions: ",
                      "A newspaper reports that a new film has taken {t} million dollars in ticket sales. "
                      "How many million dollars in ticket sales will it take in total? Answer in millions of "
                      "dollars: ",
                      "A studio's movie is still in theaters and has earned {t} million dollars. What will it "
                      "earn at the box office in total? Total millions: ",
                      "A film is {t} million dollars into its theatrical run. How many million dollars will "
                      "it bring in over its whole run? Total millions: "],
            "prior": ["Think of a movie that was released in theaters. How many million dollars did it make at "
                      "the box office in total? Answer in millions of dollars: ",
                      "A film has finished its run in theaters. What was its total gross? Total millions: ",
                      "A newspaper reports on a film that has left theaters. How many million dollars in "
                      "ticket sales did it take in total? Answer in millions of dollars: ",
                      "A studio's movie has finished its run. What did it earn at the box office in total? "
                      "Total millions: ",
                      "A film's theatrical run is over. How many million dollars did it bring in over its "
                      "whole run? Total millions: "]}},
    "pharaohs": {
        "family": "erlang", "prior": {"beta": 9.34},                                   # G&T
        "t": range(1, 41), "R_max": 150,
        "templates": {
            "remaining": ["A history of ancient Egypt notes that at one point a pharaoh had already ruled for "
                          "{t} years. For how many more years did he rule? Answer in years: ",
                          "A pharaoh has been on the throne for {t} years. How many more years will his reign "
                          "last? Years left: ",
                          "An inscription is dated to year {t} of a pharaoh's reign. For how many more years did "
                          "he reign after that? Answer in years: ",
                          "An Egyptian king has ruled for {t} years so far. How many more years will he remain "
                          "king? Years left: ",
                          "Archaeologists find a tomb painting made when a pharaoh had reigned for {t} years. "
                          "How many years of his reign were still to come? Remaining years: "],
            "total": ["A history of ancient Egypt notes that at one point a pharaoh had already ruled for "
                      "{t} years. How long did his reign last in total? Answer in years: ",
                      "A pharaoh has been on the throne for {t} years. How many years will his reign last "
                      "in total? Total years: ",
                      "An inscription is dated to year {t} of a pharaoh's reign. How many years did his whole "
                      "reign last? Answer in years: ",
                      "An Egyptian king has ruled for {t} years so far. How many years will he be king in "
                      "total? Total years: ",
                      "Archaeologists find a tomb painting made when a pharaoh had reigned for {t} years. How "
                      "long was his reign in total? Total years: "],
            "prior": ["A history of ancient Egypt lists the pharaohs and their reigns. How many years did one "
                      "pharaoh's reign last in total? Answer in years: ",
                      "Think of a pharaoh of ancient Egypt. How many years did his reign last in total? "
                      "Total years: ",
                      "An inscription names a pharaoh. How many years did his whole reign last? "
                      "Answer in years: ",
                      "Think of an Egyptian king. How many years was he king in total? Total years: ",
                      "Archaeologists find the tomb of a pharaoh. How long was his reign in total? "
                      "Total years: "]}},
    "house": {
        "family": "erlang", "prior": {"beta": 5.46},                                   # refit, fit_everyday_priors.py
        "t": range(1, 36), "R_max": 100,
        "templates": {
            "remaining": ["A member of the House of Representatives has served for {t} years so far. How many "
                          "more years will they serve? Answer in years: ",
                          "A congressman has been in the House for {t} years. How many more years will he stay "
                          "in the House? Years left: ",
                          "A U.S. Representative first took office {t} years ago and is still serving. How many more years will "
                          "they stay in the House? Answer in years: ",
                          "A congresswoman has represented her district for {t} years. How many more years "
                          "will she keep her seat? Years left: ",
                          "A House member has held office for {t} years. How many more years until they leave "
                          "the House? Remaining years: "],
            "total": ["A member of the House of Representatives has served for {t} years so far. How many "
                      "years will they serve in total? Answer in years: ",
                      "A congressman has been in the House for {t} years. How many years will he spend in "
                      "the House in total? Total years: ",
                      "A U.S. Representative first took office {t} years ago and is still serving. How many years will they "
                      "serve in the House in total? Answer in years: ",
                      "A congresswoman has represented her district for {t} years. How many years will she "
                      "hold her seat in total? Total years: ",
                      "A House member has held office for {t} years. How long will their whole time in the "
                      "House be? Total years: "],
            "prior": ["Think of a member of the House of Representatives who has left office. How many years "
                      "did they serve in the House in total? Answer in years: ",
                      "A congressman has retired from the House. How many years did he spend in the House in "
                      "total? Total years: ",
                      "A former U.S. Representative has left Congress. How many years did they serve in the "
                      "House in total? Answer in years: ",
                      "A congresswoman has given up her seat. How many years did she hold it in total? "
                      "Total years: ",
                      "A House member has left office. How long was their whole time in the House? "
                      "Total years: "]}},
    "pregnancy": {
        "family": "gaussian", "prior": {"mu": 39.11, "sigma": 1.57},                 # refit, fit_everyday_priors.py
        "t": range(4, 43), "R_max": 45,
        "templates": {
            "remaining": ["A woman is {t} weeks pregnant. How many more weeks until she gives birth? Answer in weeks: ",
                          "My sister is {t} weeks into her pregnancy. How many more weeks until the baby arrives? "
                          "Weeks left: ",
                          "A midwife sees a patient who is {t} weeks pregnant. How many weeks does she have left "
                          "before delivery? Answer in weeks: ",
                          "An expecting mother has been pregnant for {t} weeks. How many more weeks will her "
                          "pregnancy last? Weeks left: ",
                          "A pregnancy has lasted {t} weeks so far. How many more weeks until the birth? "
                          "Remaining weeks: "],
            "total": ["A woman is {t} weeks pregnant. How many weeks will her pregnancy last in total? "
                      "Answer in weeks: ",
                      "My sister is {t} weeks into her pregnancy. How long will her whole pregnancy be? "
                      "Total weeks: ",
                      "A midwife sees a patient who is {t} weeks pregnant. At how many weeks will she deliver? "
                      "Answer in weeks: ",
                      "An expecting mother has been pregnant for {t} weeks. How many weeks will her pregnancy last "
                      "in total? Total weeks: ",
                      "A pregnancy has lasted {t} weeks so far. How many weeks long will it be in total? "
                      "Total weeks: "],
            "prior": ["A woman has just given birth. How many weeks did her pregnancy last in total? "
                      "Answer in weeks: ",
                      "My sister just had her baby. How long was her whole pregnancy? Total weeks: ",
                      "A midwife delivers a baby. At how many weeks was the baby born? Answer in weeks: ",
                      "A new mother looks back on her pregnancy. How many weeks did it last in total? "
                      "Total weeks: ",
                      "A pregnancy has ended in a birth. How many weeks long was it in total? Total weeks: "]}},
    # Memoryless by mechanism: after t rolls with no six, the rolls left (counting the six) are geometric with
    # p = 1/6 for every t. That is the Erlang kernel exp(-R / beta) with beta = 1 / ln(6/5), on R >= 1.
    "dice": {
        "family": "erlang", "prior": {"beta": 1 / np.log(6 / 5)},                     # exact
        "t": range(1, 41), "R_min": 1, "R_max": 60,
        "templates": {
            "remaining": ["Someone has rolled a fair die {t} times in a row without getting a six. How many more "
                          "rolls until they get a six? Answer in rolls: ",
                          "You keep rolling a die until a six comes up. You have rolled {t} times so far with no six. "
                          "How many more rolls do you expect to need? Rolls left: ",
                          "In a board game you need a six to start. You have thrown the die {t} times without one. "
                          "How many more throws until you get it? Answer in throws: ",
                          "A child rolls a die over and over, waiting for a six. After {t} rolls there has been no "
                          "six. How many more rolls will it take? Rolls left: ",
                          "A gambler has thrown a die {t} times and has not seen a six. How many more throws until "
                          "the first six? Remaining throws: "],
            "total": ["Someone has rolled a fair die {t} times in a row without getting a six. How many rolls will "
                      "it take in total to get a six? Answer in rolls: ",
                      "You keep rolling a die until a six comes up. You have rolled {t} times so far with no six. "
                      "How many rolls will you need in total? Total rolls: ",
                      "In a board game you need a six to start. You have thrown the die {t} times without one. How "
                      "many throws will it take in total? Answer in throws: ",
                      "A child rolls a die over and over, waiting for a six. After {t} rolls there has been no six. "
                      "How many rolls will it take altogether? Total rolls: ",
                      "A gambler has thrown a die {t} times and has not seen a six. On which throw will the first "
                      "six come? Throw number: "],
            "prior": ["Someone rolled a fair die until they got a six. How many rolls did it take in total? "
                      "Answer in rolls: ",
                      "You rolled a die until a six came up. How many rolls did you need in total? Total rolls: ",
                      "In a board game you needed a six to start. How many throws did it take in total? "
                      "Answer in throws: ",
                      "A child rolled a die over and over until a six came up. How many rolls did it take "
                      "altogether? Total rolls: ",
                      "A gambler threw a die until the first six. On which throw did it come? Throw number: "]}},
    # No public source for how long books stay in print: only Llama's own conditional predicts the code.
    "books": {
        "family": None, "prior": None,
        "t": range(1, 101, 3), "R_max": 500,
        "templates": {
            "remaining": ["A book has been in print for {t} years. How many more years will it stay in print? "
                          "Answer in years: ",
                          "A novel was first published {t} years ago and people still buy it. For how many more "
                          "years will it be in print? Years left: ",
                          "A publisher has kept a title on sale for {t} years. How many more years will it keep "
                          "selling it? Answer in years: ",
                          "A bookshop has stocked the same classic for {t} years. For how many more years will "
                          "it stay in print? Years left: ",
                          "A cookbook has been in print for {t} years. How many more years until it goes out of "
                          "print? Remaining years: "],
            "total": ["A book has been in print for {t} years. How many years will it stay in print in total? "
                      "Answer in years: ",
                      "A novel was first published {t} years ago and people still buy it. How many years will it "
                      "be in print altogether? Total years: ",
                      "A publisher has kept a title on sale for {t} years. How many years will it sell it in "
                      "total? Answer in years: ",
                      "A bookshop has stocked the same classic for {t} years. How many years will it be in print "
                      "in total? Total years: ",
                      "A cookbook has been in print for {t} years. How long will its whole time in print be? "
                      "Total years: "],
            "prior": ["A book has gone out of print. How many years was it in print in total? Answer in years: ",
                      "A novel is no longer sold. How many years was it in print altogether? Total years: ",
                      "A publisher has stopped selling a title. How many years did it sell it in total? "
                      "Answer in years: ",
                      "A classic finally went out of print. How many years was it in print in total? "
                      "Total years: ",
                      "A cookbook went out of print. How long was its whole time in print? Total years: "]}},
    # Publicly traded firms have an age-independent hazard (Daepp et al. 2015): memoryless, like the pharaohs.
    "company": {
        "family": "erlang", "prior": {"beta": 15.1},                                   # Daepp et al., fit_everyday_priors.py
        "t": range(1, 101, 3), "R_max": 300,
        "templates": {
            "remaining": ["A company has been in business for {t} years. How many more years will it stay in "
                          "business? Answer in years: ",
                          "A family firm was founded {t} years ago and is still trading. How many more years will "
                          "it last? Years left: ",
                          "A listed company went public {t} years ago and still exists. How many more years until "
                          "it disappears? Answer in years: ",
                          "A local business has been open for {t} years. For how many more years will it stay "
                          "open? Years left: ",
                          "A manufacturer has been operating for {t} years. How many more years will it keep "
                          "operating? Remaining years: "],
            "total": ["A company has been in business for {t} years. How many years will it be in business in "
                      "total? Answer in years: ",
                      "A family firm was founded {t} years ago and is still trading. How many years will it last "
                      "altogether? Total years: ",
                      "A listed company went public {t} years ago and still exists. How many years will it exist "
                      "in total? Answer in years: ",
                      "A local business has been open for {t} years. How many years will it be open in total? "
                      "Total years: ",
                      "A manufacturer has been operating for {t} years. How long will it operate in total? "
                      "Total years: "],
            "prior": ["A company has gone out of business. How many years was it in business in total? "
                      "Answer in years: ",
                      "A family firm has closed down. How many years did it last altogether? Total years: ",
                      "A listed company has disappeared. How many years did it exist in total? Answer in years: ",
                      "A local business has closed for good. How many years was it open in total? Total years: ",
                      "A manufacturer has stopped operating. How long did it operate in total? Total years: "]}},
}


def log_prior(spec, T):
    """Unnormalized log prior of the total ``T`` under a domain spec's ``family`` and ``prior``."""
    family, p = spec["family"], spec["prior"]
    if family == "gaussian":
        return -(T - p["mu"]) ** 2 / (2 * p["sigma"] ** 2)
    if family == "power":
        return -p["gamma"] * np.log(T)
    if family == "erlang":                                                 # shape 2
        return np.log(T) - T / p["beta"]
    raise ValueError(f"no world prior for family {family!r}")


def remaining_logcond(spec):
    """``(t, R, L)`` for a domain spec (an entry of DOMAINS, or one with another family's prior swapped in):
    ``L[i, j] = log p(R_j | t_i)`` normalized over the grid ``R = R_min..R_max`` (R_min 0 unless the spec sets it)."""
    t, R = np.array(spec["t"]), np.arange(spec.get("R_min", 0), spec["R_max"] + 1)
    T = t[:, None] + R[None, :]
    L = log_prior(spec, T) - np.log(T)
    return t, R, L - logsumexp(L, 1, keepdims=True)


def prompts(domain, question):
    """One list per template of ``(t, text)`` over the domain's t grid; ``question`` is 'remaining' or 'total'."""
    return [[(t, PREFIX[question] + tpl.format(t=t)) for t in DOMAINS[domain]["t"]]
            for tpl in DOMAINS[domain]["templates"][question]]


def prior_prompts(domain):
    """The t-free questions eliciting the model's prior over totals, after the total-question examples."""
    return [PREFIX["total"] + tpl for tpl in DOMAINS[domain]["templates"]["prior"]]


def fit_prior(q):
    """Weighted maximum likelihood of each family on ``q``, a distribution over the totals 1..len(q).
    Returns ``(best family, {family: {"family", "prior", "loglik"}})``; a fit swapped into a domain spec
    gives that family's kernel."""
    T = np.arange(1, len(q) + 1); q = q / q.sum()
    m = (q * T).sum(); sd = np.sqrt((q * (T - m) ** 2).sum())
    init = {"gaussian": {"mu": m, "sigma": sd}, "power": {"gamma": 1.5}, "erlang": {"beta": m / 2}}
    fits = {}
    for f, p0 in init.items():
        def loglik(x):
            lp = log_prior({"family": f, "prior": dict(zip(p0, np.exp(x)))}, T)
            return (q * (lp - logsumexp(lp))).sum()
        x = minimize(lambda x: -loglik(x), np.log(list(p0.values())), method="Nelder-Mead",
                     options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000}).x
        fits[f] = {"family": f, "prior": dict(zip(p0, np.exp(x))), "loglik": loglik(x)}
    return max(fits, key=lambda f: fits[f]["loglik"]), fits


def number_tokens(tok):
    """Token ids of '0'..'999', each a single token."""
    out = []
    for n in range(1000):
        (i,) = tok.encode(str(n), add_special_tokens=False)
        out.append(i)
    return out


def encode(tok, text):
    """``(ids, at)``: BOS plus the prompt's token ids, and the index of the token holding t (the only digits
    on the prompt's last line, the question)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    line = text.rindex("\n") + 1
    s = line + re.search(r"\d+", text[line:]).start()
    at = 1 + next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= s < b)
    return [tok.bos_token_id] + enc["input_ids"], at
