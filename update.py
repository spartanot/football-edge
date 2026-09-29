"""Football Edge data updater.

Downloads the latest nflverse schedule/odds and play-by-play data, rebuilds the
prediction model, and writes data.json for the app. Run by GitHub Actions daily.
"""
import json, math, datetime as dt
import numpy as np, pandas as pd
from scipy.optimize import minimize

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
PBP_COLS = ["game_id", "posteam", "epa", "wp", "qb_dropback", "passer_player_id", "play_type"]
RENAME = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
NICK = dict(ARI="Cardinals", ATL="Falcons", BAL="Ravens", BUF="Bills", CAR="Panthers", CHI="Bears",
            CIN="Bengals", CLE="Browns", DAL="Cowboys", DEN="Broncos", DET="Lions", GB="Packers",
            HOU="Texans", IND="Colts", JAX="Jaguars", KC="Chiefs", LA="Rams", LAC="Chargers",
            LV="Raiders", MIA="Dolphins", MIN="Vikings", NE="Patriots", NO="Saints", NYG="Giants",
            NYJ="Jets", PHI="Eagles", PIT="Steelers", SEA="Seahawks", SF="49ers", TB="Buccaneers",
            TEN="Titans", WAS="Commanders")
ELO = dict(K=22.11, hfa=54.5, rev=0.469)                        # tuned on 2003-2019
FEAT = dict(k0=75, dec=0.97, a=0.1, rev=0.5, sdec=0.8)          # tuned on 2022-2024
L2 = 0.01
FIRST_PBP = 2020


def load_games():
    return pd.read_csv(GAMES_URL)


def load_pbp(years):
    out = []
    for y in years:
        try:
            out.append(pd.read_parquet(PBP_URL.format(y), columns=PBP_COLS))
            print("pbp", y, "ok")
        except Exception as e:  # current season file may not exist yet
            print("pbp", y, "skipped:", e)
    return pd.concat(out, ignore_index=True)


def dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def run(games=None, pbp=None, out="data.json"):
    df = load_games() if games is None else games.copy()
    for c in ("home_team", "away_team"):
        df[c] = df[c].replace(RENAME)
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    season = int(df.season.max())
    played = df.home_score.notna().values
    H, A, S = df.home_team.values, df.away_team.values, df.season.values
    hs, as_ = df.home_score.values, df.away_score.values
    neutral = (df.location == "Neutral").values
    y = np.where(hs > as_, 1.0, np.where(hs < as_, 0.0, np.nan))

    # Elo
    R, pt, cur = {}, np.zeros(len(df)), None
    for i in range(len(df)):
        if S[i] != cur:
            for t in R: R[t] = 1500 + (R[t] - 1500) * (1 - ELO["rev"])
            cur = S[i]
        rh, ra = R.get(H[i], 1500), R.get(A[i], 1500)
        d = rh - ra + (0 if neutral[i] else ELO["hfa"])
        pt[i] = 1 / (1 + 10 ** (-d / 400))
        if played[i] and hs[i] != as_[i]:
            hw = hs[i] > as_[i]; wd = d if hw else -d
            mult = math.log(abs(hs[i] - as_[i]) + 1) * 2.2 / (wd * 0.001 + 2.2)
            delta = ELO["K"] * mult * ((1.0 if hw else 0.0) - pt[i])
            R[H[i]], R[A[i]] = rh + delta, ra - delta

    # play-by-play ratings
    P = load_pbp(range(FIRST_PBP, season + 1)) if pbp is None else pbp.copy()
    P["posteam"] = P.posteam.replace(RENAME)
    P = P[P.play_type.isin(["pass", "run"]) & P.epa.notna() & P.wp.between(.05, .95) & P.posteam.notna()]
    R_ = P[P.play_type == "run"].groupby(["game_id", "posteam"]).epa.agg(["sum", "count"])
    D_ = P[P.qb_dropback == 1]
    T_ = D_.groupby(["game_id", "posteam"]).epa.agg(["sum", "count"])
    Q_ = D_[D_.passer_player_id.notna()].groupby(["game_id", "posteam", "passer_player_id"]).epa.agg(["sum", "count"]).reset_index()
    rush = {k: (r["sum"], r["count"]) for k, r in R_.iterrows()}
    tdb = {k: (r["sum"], r["count"]) for k, r in T_.iterrows()}
    qbo = {}
    for g_, t_, p_, s_, c_ in zip(Q_.game_id, Q_.posteam, Q_.passer_player_id, Q_["sum"], Q_["count"]):
        qbo.setdefault((g_, t_), []).append((p_, s_, c_))
    lgr = R_["sum"].sum() / R_["count"].sum(); lgp = T_["sum"].sum() / T_["count"].sum()
    GID, HQ, AQ = df.game_id.values, df.home_qb_id.values, df.away_qb_id.values
    k0, dk, a, rev, sdec = FEAT["k0"], FEAT["dec"], FEAT["a"], FEAT["rev"], FEAT["sdec"]
    qs, offr, defr, pdf = {}, {}, {}, {}
    F = np.full((len(df), 3), np.nan); cur = None
    rate = lambda pid: (qs[pid][0] + k0 * lgp) / (qs[pid][1] + k0) if pid in qs else lgp
    for i in range(len(df)):
        if S[i] < FIRST_PBP: continue
        if S[i] != cur:
            for t in offr: offr[t] *= (1 - rev); defr[t] *= (1 - rev); pdf[t] *= (1 - rev)
            for p in qs: qs[p][0] *= sdec; qs[p][1] *= sdec
            cur = S[i]
        h, a_, g = H[i], A[i], GID[i]
        qh, qa = rate(HQ[i]), rate(AQ[i])
        rh = lgr + offr.get(h, 0) + defr.get(a_, 0); ra = lgr + offr.get(a_, 0) + defr.get(h, 0)
        F[i] = (qh - qa, rh - ra, pdf.get(a_, 0) - pdf.get(h, 0))
        if not played[i]: continue
        for team, opp in ((h, a_), (a_, h)):
            ob = rush.get((g, team))
            if ob:
                e = ob[0] / ob[1] - (lgr + offr.get(team, 0) + defr.get(opp, 0))
                offr[team] = offr.get(team, 0) + a * e; defr[opp] = defr.get(opp, 0) + a * e
            tb = tdb.get((g, team))
            if tb:
                qt = qh if team == h else qa
                e = tb[0] / tb[1] - (qt + pdf.get(opp, 0))
                pdf[opp] = pdf.get(opp, 0) + a * e
            for pid, sm, ct in qbo.get((g, team), []):
                s_ = qs.setdefault(pid, [0.0, 0.0]); s_[0] = s_[0] * dk + sm; s_[1] = s_[1] * dk + ct

    # model: logistic regression on standardized features
    hm, am = df.home_moneyline.values, df.away_moneyline.values
    base = ~np.isnan(hm) & ~np.isnan(am) & played & (S >= FIRST_PBP + 1)
    elo_l = np.log(np.clip(pt, .01, .99) / (1 - np.clip(pt, .01, .99)))
    X = np.column_stack([np.nan_to_num(F), elo_l, np.where(neutral, 0.0, 1.0)])
    sd = X[base].std(0); sd[-1] = 1; X = X / sd

    def fit(mask):
        Xt, yt = X[mask], y[mask]
        f = lambda w: np.mean(np.log1p(np.exp(-(Xt @ w) * (2 * yt - 1)))) + L2 * np.sum(w[:-1] ** 2)
        return minimize(f, np.zeros(X.shape[1]), method="BFGS").x
    sig = lambda z: 1 / (1 + np.exp(-z))
    p_past = sig(X @ fit(base & (S < season) & ~np.isnan(y)))   # out of sample for this season
    p_now = sig(X @ fit(base & ~np.isnan(y)))                     # uses every finished game
    kp = np.full(len(df), np.nan)
    ok = ~np.isnan(hm) & ~np.isnan(am)
    for i in np.where(ok)[0]:
        iH, iA = 1 / dec(hm[i]), 1 / dec(am[i]); kp[i] = iH / (iH + iA)

    cs = (S == season)
    games, past = [], []
    for i in np.where(cs & ok)[0]:
        g = dict(id=GID[i], d=df.gameday.values[i], t=str(df.gametime.values[i]), wk=int(df.week.values[i]),
                 a=A[i], h=H[i], am=int(am[i]), hm=int(hm[i]),
                 aq=str(df.away_qb_name.values[i]), hq=str(df.home_qb_name.values[i]),
                 kp=round(float(kp[i]), 3), mp=round(float(p_now[i]), 3))
        extra = dict(sp=df.spread_line.values[i], aso=df.away_spread_odds.values[i], hso=df.home_spread_odds.values[i],
                     tot=df.total_line.values[i], ov=df.over_odds.values[i], un=df.under_odds.values[i])
        for k, v in extra.items():
            g[k] = None if pd.isna(v) else (float(v) if k in ("sp", "tot") else int(v))
        if played[i]:
            g["sa"], g["sh"] = int(as_[i]), int(hs[i])
            past.append([season, int(df.week.values[i]), str(df.game_type.values[i]), df.gameday.values[i], A[i], H[i],
                         int(as_[i]), int(hs[i]), int(am[i]), int(hm[i]), round(float(kp[i]), 3), round(float(p_past[i]), 3)])
        games.append(g)

    rec, hist = {}, []
    for i in np.where(cs & played & (df.game_type.values == "REG"))[0]:
        for t, won in ((H[i], hs[i] > as_[i]), (A[i], as_[i] > hs[i])):
            r = rec.setdefault(t, [0, 0])
            if hs[i] != as_[i]: r[0 if won else 1] += 1
        hist.append(f"{NICK[H[i]]},{NICK[A[i]]},{int(hs[i])},{int(as_[i])}")
    teams = ",".join(f"{NICK[t]}:{w}-{l}" for t, (w, l) in sorted(rec.items()))

    decided = [r for r in past if r[6] != r[7]]
    val = [(r, sd_, o) for r in past for sd_, p, o in (("h", r[11], r[9]), ("a", 1 - r[11], r[8])) if p * dec(o) - 1 >= .03]
    stats = dict(n=len(decided), book=sum((r[10] >= .5) == (r[7] > r[6]) for r in decided),
                 model=sum((r[11] >= .5) == (r[7] > r[6]) for r in decided),
                 vn=len(val), vw=sum(r[6] != r[7] and ((r[7] > r[6]) == (s_ == "h")) for r, s_, o in val),
                 vdog=sum(o > 0 for _, _, o in val), weeks=sorted({r[1] for r in past}))
    data = dict(updated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), season=season,
                games=games, past=past, teams=teams, hist=";".join(hist), stats=stats)
    with open(out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", out, "| games", len(games), "| finished", len(past))
    return data


if __name__ == "__main__":
    run()
