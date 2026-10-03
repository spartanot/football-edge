"""College football data updater for Football Edge.

Uses the CollegeFootballData.com API (key in the CFBD_API_KEY secret).
Past seasons are cached in cfb_cache/ so each run only downloads the current
season (3 API calls). Scheduled runs only refresh twice a day to stay well
inside the free plan's monthly request limit.
"""
import os, sys, json, math, datetime as dt, urllib.request, urllib.parse
import numpy as np
from scipy.optimize import minimize

API = "https://api.collegefootballdata.com"
CACHE = "cfb_cache"
OUT = "cfb_data.json"
FIRST = 2021                      # first season used to build ratings
ELO = dict(K=25.0, hfa=60.0, rev=0.33)
RATE = dict(a=0.08, rev=0.5)
L2 = 0.01
REFRESH_HOURS_UTC = (10, 22)      # scheduled refreshes (6 AM / 6 PM Eastern)
BOOKS = ("DraftKings", "ESPN Bet", "FanDuel", "Bovada", "William Hill (New Jersey)", "consensus")


def get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def num(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def api(path, **params):
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        sys.exit("CFBD_API_KEY is not set")
    url = API + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def fetch_season(year, refresh):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for name, path, params in (("games", "/games", dict(year=year, seasonType="both")),
                               ("lines", "/lines", dict(year=year, seasonType="both")),
                               ("ppa", "/ppa/games", dict(year=year, excludeGarbageTime="true"))):
        f = os.path.join(CACHE, f"{year}_{name}.json")
        if refresh or not os.path.exists(f):
            data = api(path, **params)
            with open(f, "w") as fh:
                json.dump(data, fh)
            print("downloaded", year, name, len(data))
        with open(f) as fh:
            out[name] = json.load(fh)
    return out


def dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def pick_line(lines):
    best = None
    for pref in BOOKS + (None,):
        for ln in lines or []:
            if pref is not None and get(ln, "provider") != pref:
                continue
            hm, am = num(get(ln, "homeMoneyline", "home_moneyline")), num(get(ln, "awayMoneyline", "away_moneyline"))
            sp = num(get(ln, "spread"))
            if hm is not None and am is not None and hm != 0 and am != 0:
                return ln
            if best is None and sp is not None:
                best = ln
    return best


def run():
    now = dt.datetime.now(dt.timezone.utc)
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and now.hour not in REFRESH_HOURS_UTC and os.path.exists(OUT):
        print("College data refreshes twice a day; skipping this run.")
        return
    season = now.year if now.month >= 8 else now.year - 1
    raw = {y: fetch_season(y, refresh=(y == season)) for y in range(FIRST, season + 1)}

    # ---- assemble games ----
    G = []
    for y, d in raw.items():
        lines = {str(get(x, "id")): x for x in d["lines"]}
        for g in d["games"]:
            gid = str(get(g, "id"))
            hp, ap = num(get(g, "homePoints", "home_points")), num(get(g, "awayPoints", "away_points"))
            start = get(g, "startDate", "start_date", default="")
            try:
                st = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError:
                continue
            et = st - dt.timedelta(hours=4 if 3 <= st.month <= 10 else 5)
            ln = pick_line(get(lines.get(gid, {}), "lines", default=[]))
            G.append(dict(
                id=gid, season=int(get(g, "season", default=y)), week=int(get(g, "week", default=0)),
                stype=get(g, "seasonType", "season_type", default="regular"), st=st, d=et.strftime("%Y-%m-%d"),
                t="TBD" if get(g, "startTimeTBD", "start_time_tbd") else et.strftime("%H:%M"),
                h=get(g, "homeTeam", "home_team"), a=get(g, "awayTeam", "away_team"),
                hc=get(g, "homeConference", "home_conference", default="") or "",
                ac=get(g, "awayConference", "away_conference", default="") or "",
                hcl=(get(g, "homeClassification", "homeDivision", "home_division", default="") or "").lower(),
                acl=(get(g, "awayClassification", "awayDivision", "away_division", default="") or "").lower(),
                neutral=bool(get(g, "neutralSite", "neutral_site", default=False)),
                done=hp is not None and ap is not None and bool(get(g, "completed", default=True)),
                hs=hp, as_=ap,
                hm=num(get(ln, "homeMoneyline", "home_moneyline")) if ln else None,
                am=num(get(ln, "awayMoneyline", "away_moneyline")) if ln else None,
                spread=num(get(ln, "spread")) if ln else None,
                tot=num(get(ln, "overUnder", "over_under")) if ln else None))
    G = [g for g in G if g["h"] and g["a"]]
    G.sort(key=lambda g: (g["st"], g["id"]))
    for g in G:
        if g["hm"] == 0 or g["am"] == 0:
            g["hm"] = g["am"] = None
    n = len(G)
    S = np.array([g["season"] for g in G])
    done = np.array([g["done"] for g in G])
    y = np.array([np.nan if not g["done"] or g["hs"] == g["as_"] else float(g["hs"] > g["as_"]) for g in G])
    neutral = np.array([g["neutral"] for g in G])

    # ---- Elo (FCS and lower start below FBS) ----
    base_r = lambda cl: 1500.0 if cl == "fbs" else 1250.0
    R, cls, pt, cur = {}, {}, np.zeros(n), None
    for i, g in enumerate(G):
        cls.setdefault(g["h"], g["hcl"]); cls.setdefault(g["a"], g["acl"])
        if g["season"] != cur:
            for t in R: R[t] = base_r(cls.get(t)) + (R[t] - base_r(cls.get(t))) * (1 - ELO["rev"])
            cur = g["season"]
        rh, ra = R.get(g["h"], base_r(g["hcl"])), R.get(g["a"], base_r(g["acl"]))
        dd = rh - ra + (0 if g["neutral"] else ELO["hfa"])
        pt[i] = 1 / (1 + 10 ** (-dd / 400))
        if g["done"] and g["hs"] != g["as_"]:
            hw = g["hs"] > g["as_"]; wd = dd if hw else -dd
            mult = math.log(abs(g["hs"] - g["as_"]) + 1) * 2.2 / (wd * 0.001 + 2.2)
            delta = ELO["K"] * mult * ((1.0 if hw else 0.0) - pt[i])
            R[g["h"]], R[g["a"]] = rh + delta, ra - delta

    # ---- opponent-adjusted PPA (college EPA) ratings ----
    obs = {}
    for yv, d in raw.items():
        for p in d["ppa"]:
            off = get(p, "offense", default={}) or {}
            ps, rs = num(get(off, "passing")), num(get(off, "rushing"))
            if ps is not None and rs is not None:
                obs[(str(get(p, "gameId", "game_id")), get(p, "team"))] = (ps, rs)
    lp = np.mean([v[0] for v in obs.values()]) if obs else 0.0
    lr = np.mean([v[1] for v in obs.values()]) if obs else 0.0
    op, dp, orr, dr = {}, {}, {}, {}
    F = np.zeros((n, 2)); cur = None; a = RATE["a"]
    for i, g in enumerate(G):
        if g["season"] != cur:
            for dct in (op, dp, orr, dr):
                for t in dct: dct[t] *= (1 - RATE["rev"])
            cur = g["season"]
        h, aw = g["h"], g["a"]
        F[i] = ((op.get(h, 0) + dp.get(aw, 0)) - (op.get(aw, 0) + dp.get(h, 0)),
                (orr.get(h, 0) + dr.get(aw, 0)) - (orr.get(aw, 0) + dr.get(h, 0)))
        if not g["done"]:
            continue
        for team, opp in ((h, aw), (aw, h)):
            ob = obs.get((g["id"], team))
            if not ob:
                continue
            e = ob[0] - (lp + op.get(team, 0) + dp.get(opp, 0)); op[team] = op.get(team, 0) + a * e; dp[opp] = dp.get(opp, 0) + a * e
            e = ob[1] - (lr + orr.get(team, 0) + dr.get(opp, 0)); orr[team] = orr.get(team, 0) + a * e; dr[opp] = dr.get(opp, 0) + a * e

    # ---- market probability and model ----
    hm = np.array([np.nan if g["hm"] is None else g["hm"] for g in G])
    am = np.array([np.nan if g["am"] is None else g["am"] for g in G])
    hasml = ~np.isnan(hm) & ~np.isnan(am)
    kp = np.full(n, np.nan)
    for i, g in enumerate(G):
        if hasml[i]:
            iH, iA = 1 / dec(hm[i]), 1 / dec(am[i]); kp[i] = iH / (iH + iA)
        elif g["spread"] is not None:
            kp[i] = 0.5 * (1 + math.erf((-g["spread"] / 15.0) / math.sqrt(2)))
    elo_l = np.log(np.clip(pt, .01, .99) / (1 - np.clip(pt, .01, .99)))
    X = np.column_stack([F, elo_l, np.where(neutral, 0.0, 1.0)])
    base = hasml & done & ~np.isnan(y) & (S >= FIRST + 1)
    sd = X[base].std(0) if base.sum() > 10 else np.ones(X.shape[1])
    sd[sd == 0] = 1; sd[-1] = 1; X = X / sd
    sig = lambda z: 1 / (1 + np.exp(-z))

    def fit(mask):
        if mask.sum() < 30:
            return np.array([0, 0, 1.0, 0.3])
        Xt, yt = X[mask], y[mask]
        f = lambda w: np.mean(np.log1p(np.exp(-(Xt @ w) * (2 * yt - 1)))) + L2 * np.sum(w[:-1] ** 2)
        return minimize(f, np.zeros(X.shape[1]), method="BFGS").x
    p_past = sig(X @ fit(base & (S < season)))
    p_now = sig(X @ fit(base))

    # ---- output ----
    games, past = [], []
    for i, g in enumerate(G):
        if g["season"] != season or np.isnan(kp[i]):
            continue
        sp = None if g["spread"] is None else -g["spread"]          # app convention: positive = home favored
        o = dict(id="cfb" + g["id"], d=g["d"], t=g["t"], wk=g["week"], a=g["a"], h=g["h"], ac=g["ac"], hc=g["hc"],
                 am=None if np.isnan(am[i]) else int(am[i]), hm=None if np.isnan(hm[i]) else int(hm[i]),
                 sp=sp, aso=-110 if sp is not None else None, hso=-110 if sp is not None else None,
                 tot=g["tot"], ov=-110 if g["tot"] is not None else None, un=-110 if g["tot"] is not None else None,
                 aq="", hq="", kp=round(float(kp[i]), 3), mp=round(float(p_now[i]), 3), ns=bool(g["neutral"]))
        if g["done"]:
            o["sa"], o["sh"] = int(g["as_"]), int(g["hs"])
            if hasml[i]:
                past.append([season, g["week"], "REG" if g["stype"] == "regular" else "POST", g["d"], g["a"], g["h"],
                             int(g["as_"]), int(g["hs"]), int(am[i]), int(hm[i]), round(float(kp[i]), 3), round(float(p_past[i]), 3)])
        games.append(o)

    decided = [r for r in past if r[6] != r[7]]
    val = [(r, s_, o) for r in past for s_, p, o in (("h", r[11], r[9]), ("a", 1 - r[11], r[8])) if p * dec(o) - 1 >= .03]
    stats = dict(n=len(decided), book=sum((r[10] >= .5) == (r[7] > r[6]) for r in decided),
                 model=sum((r[11] >= .5) == (r[7] > r[6]) for r in decided),
                 vn=len(val), vw=sum(r[6] != r[7] and ((r[7] > r[6]) == (s_ == "h")) for r, s_, o in val),
                 vdog=sum(o > 0 for _, _, o in val),
                 vprofit=round(sum(0 if r[6] == r[7] else (100 * (dec(o) - 1) if (r[7] > r[6]) == (s_ == "h") else -100)
                                   for r, s_, o in val), 2))
    bb = dict(n=0, w=0, profit=0.0)
    for wk in sorted({r[1] for r in past}):
        c = [((pk + pm) / 2, r, s_, o) for r in past if r[1] == wk
             for s_, pk, pm, o in (("h", r[10], r[11], r[9]), ("a", 1 - r[10], 1 - r[11], r[8])) if pk > .5 and pm > .5]
        for _, r, s_, o in sorted(c, key=lambda x: -x[0])[:3]:
            if r[6] == r[7]:
                continue
            won = (r[7] > r[6]) == (s_ == "h")
            bb["n"] += 1; bb["w"] += won; bb["profit"] += 100 * (dec(o) - 1) if won else -100
    bb["profit"] = round(bb["profit"], 2); stats["best"] = bb
    data = dict(updated=now.strftime("%Y-%m-%d %H:%M UTC"), season=season, games=games, past=past,
                teams="", hist="", stats=stats)
    with open(OUT, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", OUT, "| games", len(games), "| finished with moneylines", len(past))


if __name__ == "__main__":
    run()
