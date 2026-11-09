"""Everyday predictions (Griffiths & Tenenbaum 2006) asked as *remaining* time. Having seen t, predict
R = T - t. The posterior is p(R | t) ∝ p(t + R) / (t + R), with no support cut-off. The prior is
evaluated at t + R, so it enters the t-R interaction, and the prior family fixes the predicted code of
t up to scale: Gaussian, a line in t; power law, a function of log t (it depends on R / t); Erlang, a
single point (the prior's factor T cancels the 1/T likelihood, leaving one exponential for every t).

Each domain's prior parameters only set the scale of the code and the behavioral reference medians, not
its shape. Those marked G&T are from the paper; the rest are placeholders."""
import re

import numpy as np
from scipy.special import logsumexp

_INTRO = ("Each question below gives one piece of information and asks you to predict "
                       "something from it. Give your best guess as a single number.\n")

# Templates hold no digits besides {t}, and end on the space before the answer.
DOMAINS = {
    "lifespan": {
        "family": "gaussian", "prior": {"mu": 75, "sigma": 16},                       # G&T
        "t": range(10, 96), "R_max": 110,
        "templates": {
            "remaining": ["Insurance actuaries estimate how long people will live. A man is {t} years old. "
                          "How many more years will he live? Answer in years: ",
                          "You meet a man who is {t} years old. How many more years do you expect him to live? "
                          "Years left: "],
            "total": ["Insurance actuaries estimate how long people will live. A man is {t} years old. "
                      "At what age will he die? Answer in years: ",
                      "You meet a man who is {t} years old. How old do you expect him to be when he dies? "
                      "Age at death: "]}},
    "runtime": {
        "family": "gaussian", "prior": {"mu": 110, "sigma": 20},                      # placeholder
        "t": range(10, 152, 2), "R_max": 200,
        "templates": {
            "remaining": ["You drop in on a friend who has been watching a movie for {t} minutes. "
                          "How many more minutes will the movie run? Answer in minutes: ",
                          "A movie has been playing for {t} minutes. How many minutes are left until it ends? "
                          "Minutes left: "],
            "total": ["You drop in on a friend who has been watching a movie for {t} minutes. "
                      "How long is the whole movie? Answer in minutes: ",
                      "A movie has been playing for {t} minutes. How long is the movie in total? "
                      "Total minutes: "]}},
    "grosses": {
        "family": "power", "prior": {"gamma": 1.71},                                   # G&T (the 1.5x rule)
        "t": range(1, 101), "R_max": 999,
        "templates": {
            "remaining": ["A movie has made {t} million dollars at the box office so far. How many more "
                          "million dollars will it make? Answer in millions of dollars: ",
                          "So far a film has grossed {t} million dollars. How much more will it gross before "
                          "it leaves theaters? Additional millions: "],
            "total": ["A movie has made {t} million dollars at the box office so far. How many million "
                      "dollars will it make in total? Answer in millions of dollars: ",
                      "So far a film has grossed {t} million dollars. What will its total gross be when it "
                      "leaves theaters? Total millions: "]}},
    "poems": {
        "family": "power", "prior": {"gamma": 1.71},                                   # placeholder
        "t": range(2, 81), "R_max": 999,
        "templates": {
            "remaining": ["A friend reads you a line of poetry and says it is line {t} of the poem. How many "
                          "more lines does the poem have after it? Answer in lines: ",
                          "You open a poem at line {t}. How many lines of the poem come after this one? "
                          "Lines left: "],
            "total": ["A friend reads you a line of poetry and says it is line {t} of the poem. How many "
                      "lines does the whole poem have? Answer in lines: ",
                      "You open a poem at line {t}. How many lines does the poem have in total? "
                      "Total lines: "]}},
    "pharaohs": {
        "family": "erlang", "prior": {"beta": 9.34},                                   # G&T
        "t": range(1, 41), "R_max": 150,
        "templates": {
            "remaining": ["A history of ancient Egypt notes that at one point a pharaoh had already ruled for "
                          "{t} years. For how many more years did he rule? Answer in years: ",
                          "A pharaoh has been on the throne for {t} years. How many more years will his reign "
                          "last? Years left: "],
            "total": ["A history of ancient Egypt notes that at one point a pharaoh had already ruled for "
                      "{t} years. How long did his reign last in total? Answer in years: ",
                      "A pharaoh has been on the throne for {t} years. How many years will his reign last "
                      "in total? Total years: "]}},
    "house": {
        "family": "erlang", "prior": {"beta": 7.0},                                    # placeholder
        "t": range(1, 36), "R_max": 100,
        "templates": {
            "remaining": ["A member of the House of Representatives has served for {t} years so far. How many "
                          "more years will they serve? Answer in years: ",
                          "A congressman has been in the House for {t} years. How many more years will he stay "
                          "in the House? Years left: "],
            "total": ["A member of the House of Representatives has served for {t} years so far. How many "
                      "years will they serve in total? Answer in years: ",
                      "A congressman has been in the House for {t} years. How many years will he spend in "
                      "the House in total? Total years: "]}},
}


def log_prior(spec, T):
    """Unnormalized log prior of the total ``T`` under a domain spec's ``family`` and ``prior``."""
    family, p = spec["family"], spec["prior"]
    if family == "gaussian":
        return -(T - p["mu"]) ** 2 / (2 * p["sigma"] ** 2)
    if family == "power":
        return -p["gamma"] * np.log(T)
    return np.log(T) - T / p["beta"]                                       # Erlang, shape 2


def remaining_logcond(spec):
    """``(t, R, L)`` for a domain spec (an entry of DOMAINS, or one with another family's prior swapped in):
    ``L[i, j] = log p(R_j | t_i)`` normalized over the grid ``R = 0..R_max``."""
    t, R = np.array(spec["t"]), np.arange(spec["R_max"] + 1)
    T = t[:, None] + R[None, :]
    L = log_prior(spec, T) - np.log(T)
    return t, R, L - logsumexp(L, 1, keepdims=True)


def prompts(domain, question):
    """One list per template of ``(t, text)`` over the domain's t grid; ``question`` is 'remaining' or 'total'."""
    return [[(t, _INTRO + tpl.format(t=t)) for t in DOMAINS[domain]["t"]]
            for tpl in DOMAINS[domain]["templates"][question]]


def number_tokens(tok):
    """Token ids of '0'..'999', each a single token."""
    out = []
    for n in range(1000):
        (i,) = tok.encode(str(n), add_special_tokens=False)
        out.append(i)
    return out


def encode(tok, text):
    """``(ids, at)``: BOS plus the prompt's token ids, and the index of the token holding t (the text's only digits)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    s = re.search(r"\d+", text).start()
    at = 1 + next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= s < b)
    return [tok.bos_token_id] + enc["input_ids"], at
