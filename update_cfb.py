"""College football data updater for Football Edge.

Uses the CollegeFootballData.com API (key in the CFBD_API_KEY secret).
Past seasons are cached in cfb_cache/, so a normal run downloads only the
current season's games, lines, PPA and advanced stats (4 API calls).

Feature testing: when cfb_features.json is missing, or the workflow is run by
hand, the script tests each extra stat group on past seasons (walk-forward),
keeps only the groups that improve predictions on 2023-2024, reports how they
do on 2025 onward in cfb_eval.json, and saves the choice in cfb_features.json.
"""
import os, sys, json, math, datetime as dt, urllib.request, urllib.parse
import numpy as np
from scipy.optimize import minimize

API = "https://api.collegefootballdata.com"
CACHE = "cfb_cache"
OUT, EVAL, FEATS = "cfb_data.json", "cfb_eval.json", "cfb_features.json"
FIRST = 2021
ELO = dict(K=25.0, hfa=60.0, rev=0.33)
RATE = dict(a=0.08, rev=0.5)
L2 = 0.01
REFRESH_HOURS_UTC = (10, 22)
BOOKS = ("DraftKings", "ESPN Bet", "FanDuel", "Bovada", "William Hill (New Jersey)", "consensus")
DEV, TEST_FROM = (2023, 2024), 2025
GROUPS = ["sp_prev", "returning", "talent", "portal", "coach", "line_move", "advanced"]
failed = []


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


def cached(name, yr, path, refresh=False, required=False, **params):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{yr}_{name}.json")
    if refresh or not os.path.exists(f):
        try:
            data = api(path, **params)
        except Exception as e:
            if required:
                raise
            print("could not download", yr, name, "-", e)
            failed.append(f"{yr} {name}: {e}")
            if os.path.exists(f):
                return json.load(open(f))
            json.dump([], open(f, "w"))   # don't spend API calls retrying every run; delete the file to retry
            return []
        json.dump(data, open(f, "w"))
        print("downloaded", yr, name, len(data))
    return json.load(open(f))


def fetch_season(y, cur):
    r = y == cur
    return dict(
        games=cached("games", y, "/games", r, True, year=y, seasonType="both"),
        lines=cached("lines", y, "/lines", r, True, year=y, seasonType="both"),
        ppa=cached("ppa", y, "/ppa/games", r, True, year=y, excludeGarbageTime="true"),
        adv=cached("advanced", y, "/stats/game/advanced", r, year=y, excludeGarbageTime="true"),
        sp_prev=cached("sp", y - 1, "/ratings/sp", year=y - 1),
        returning=cached("returning", y, "/player/returning", year=y),
        talent=cached("talent", y, "/talent", year=y),
        portal=cached("portal", y, "/player/portal", year=y),
        coaches=cached("coaches", y, "/coaches", year=y),
        coaches_prev=cached("coaches", y - 1, "/coaches", year=y - 1))


def dec(o):
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def pick_line(lines):
    best = None
    for pref in BOOKS + (None,):
        for ln in lines or []:
            if pref is not None and get(ln, "provider") != pref:
                continue
            hm, am = num(get(ln, "homeMoneyline", "home_moneyline")), num(get(ln, "awayMoneyline", "away_moneyline"))
            if hm and am:
                return ln
            if best is None and num(get(ln, "spread")) is not None:
                best = ln
    return best


def coach_map(rows, year):
    m = {}
    for c in rows or []:
        for s in get(c, "seasons", default=[]) or []:
            if int(get(s, "year", default=-1)) == year:
                m[get(s, "school")] = f"{get(c, 'firstName', 'first_name')} {get(c, 'lastName', 'last_name')}"
    return m


def season_priors(d, y):
    sp = {get(r, "team"): num(get(r, "rating")) for r in d["sp_prev"] if num(get(r, "rating")) is not None}
    ret = {get(r, "team"): num(get(r, "percentPPA", "percent_ppa")) for r in d["returning"]}
    tal = {get(r, "team", "school"): num(get(r, "talent")) for r in d["talent"]}
    portal = {}
    for p in d["portal"]:
        st = num(get(p, "stars")) or 2.0
        o, t = get(p, "origin"), get(p, "destination")
        if t: portal[t] = portal.get(t, 0) + st
        if o: portal[o] = portal.get(o, 0) - st
    now, prev = coach_map(d["coaches"], y), coach_map(d["coaches_prev"], y - 1)
    newc = {t: 1.0 for t, c in now.items() if t in prev and prev[t] != c}
    clean = lambda m: {k: v for k, v in m.items() if v is not None}
    return dict(sp_prev=clean(sp), returning=clean(ret), talent=clean(tal), portal=portal, coach=newc)


def build(raw, season):
    G = []
    for y, d in raw.items():
        lines = {str(get(x, "id")): x for x in d["lines"]}
        for g in d["games"]:
            gid = str(get(g, "id"))
            hp, ap = num(get(g, "homePoints", "home_points")), num(get(g, "awayPoints", "away_points"))
            try:
                st = dt.datetime.fromisoformat(get(g, "startDate", "start_date", default="").replace("Z", "+00:00"))
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
                spread_open=num(get(ln, "spreadOpen", "spread_open")) if ln else None,
                tot=num(get(ln, "overUnder", "over_under")) if ln else None))
    G = [g for g in G if g["h"] and g["a"]]
    G.sort(key=lambda g: (g["st"], g["id"]))
    for g in G:
        if not g["hm"] or not g["am"]:
            g["hm"] = g["am"] = None
    return G


def features(G, raw):
    n = len(G)
    # Elo
    base_r = lambda cl: 1500.0 if cl == "fbs" else 1250.0
    R, cls, pt, cur = {}, {}, np.zeros(n), None
    for i, g in enumerate(G):
        cls.setdefault(g["h"], g["hcl"]); cls.setdefault(g["a"], g["acl"])
        if g["season"] != cur:
            for t in R: R[t] = base_r(cls.get(t)) + (R[t] - base_r(cls.get(t))) * (1 - ELO["rev"])
            cur = g["season"]
        rh, ra = R.get(g["h"], base_r(g["hcl"])), R.get(g["a"], base_r(g["acl"]))
        d = rh - ra + (0 if g["neutral"] else ELO["hfa"])
        pt[i] = 1 / (1 + 10 ** (-d / 400))
        if g["done"] and g["hs"] != g["as_"]:
            hw = g["hs"] > g["as_"]; wd = d if hw else -d
            mult = math.log(abs(g["hs"] - g["as_"]) + 1) * 2.2 / (wd * 0.001 + 2.2)
            delta = ELO["K"] * mult * ((1.0 if hw else 0.0) - pt[i])
            R[g["h"]], R[g["a"]] = rh + delta, ra - delta

    def adjusted(obs, k):
        """Opponent-adjusted offense/defense ratings for k stats, pre-game diffs."""
        lg = [np.mean([v[j] for v in obs.values()]) if obs else 0.0 for j in range(k)]
        off = [dict() for _ in range(k)]; dfn = [dict() for _ in range(k)]
        out = np.zeros((n, k)); cur = None; a = RATE["a"]
        for i, g in enumerate(G):
            if g["season"] != cur:
                for dct in off + dfn:
                    for t in dct: dct[t] *= (1 - RATE["rev"])
                cur = g["season"]
            h, aw = g["h"], g["a"]
            for j in range(k):
                out[i, j] = (off[j].get(h, 0) + dfn[j].get(aw, 0)) - (off[j].get(aw, 0) + dfn[j].get(h, 0))
            if not g["done"]:
                continue
            for team, opp in ((h, aw), (aw, h)):
                ob = obs.get((g["id"], team))
                if not ob:
                    continue
                for j in range(k):
                    e = ob[j] - (lg[j] + off[j].get(team, 0) + dfn[j].get(opp, 0))
                    off[j][team] = off[j].get(team, 0) + a * e; dfn[j][opp] = dfn[j].get(opp, 0) + a * e
        return out

    ppa, adv = {}, {}
    for d in raw.values():
        for p in d["ppa"]:
            o = get(p, "offense", default={}) or {}
            ps, rs = num(get(o, "passing")), num(get(o, "rushing"))
            if ps is not None and rs is not None:
                ppa[(str(get(p, "gameId", "game_id")), get(p, "team"))] = (ps, rs)
        for p in d["adv"]:
            o = get(p, "offense", default={}) or {}
            sr, ex = num(get(o, "successRate", "success_rate")), num(get(o, "explosiveness"))
            if sr is not None and ex is not None:
                adv[(str(get(p, "gameId", "game_id")), get(p, "team"))] = (sr, ex)
    cols = {"ppa": adjusted(ppa, 2),
            "elo": np.log(np.clip(pt, .01, .99) / (1 - np.clip(pt, .01, .99)))[:, None],
            "hfa": np.array([0.0 if g["neutral"] else 1.0 for g in G])[:, None]}
    cols["advanced"] = adjusted(adv, 2) if adv else np.zeros((n, 2))
    pri = {y: season_priors(d, y) for y, d in raw.items()}
    for grp in ("sp_prev", "returning", "talent", "portal", "coach"):
        M = np.zeros((n, 1 if grp == "coach" else 2))
        for i, g in enumerate(G):
            m = pri[g["season"]][grp]
            if not m:
                continue
            vals = list(m.values()); fill = 0.0 if grp in ("portal", "coach") else float(np.median(vals))
            dif = m.get(g["h"], fill) - m.get(g["a"], fill)
            early = max(0.0, 1 - g["week"] / 8)          # priors matter most early in the season
            M[i] = [dif] if grp == "coach" else [dif, dif * early]
        cols[grp] = M
    cols["line_move"] = np.array([[0.0 if g["spread"] is None or g["spread_open"] is None
                                   else g["spread_open"] - g["spread"]] for g in G])
    return cols


def run():
    now = dt.datetime.now(dt.timezone.utc)
    ev = os.environ.get("GITHUB_EVENT_NAME", "")
    if ev == "schedule" and now.hour not in REFRESH_HOURS_UTC and os.path.exists(OUT):
        print("College data refreshes twice a day; skipping this run.")
        return
    season = now.year if now.month >= 8 else now.year - 1
    raw = {y: fetch_season(y, season) for y in range(FIRST, season + 1)}
    G = build(raw, season)
    cols = features(G, raw)
    n = len(G)
    S = np.array([g["season"] for g in G])
    y = np.array([np.nan if not g["done"] or g["hs"] == g["as_"] else float(g["hs"] > g["as_"]) for g in G])
    hm = np.array([np.nan if g["hm"] is None else g["hm"] for g in G])
    am = np.array([np.nan if g["am"] is None else g["am"] for g in G])
    hasml = ~np.isnan(hm) & ~np.isnan(am)
    kp = np.full(n, np.nan)
    for i, g in enumerate(G):
        if hasml[i]:
            iH, iA = 1 / dec(hm[i]), 1 / dec(am[i]); kp[i] = iH / (iH + iA)
        elif g["spread"] is not None:
            kp[i] = 0.5 * (1 + math.erf((-g["spread"] / 15.0) / math.sqrt(2)))
    base = hasml & ~np.isnan(y) & (S >= FIRST + 1)
    sig = lambda z: 1 / (1 + np.exp(-z))

    def matrix(groups):
        X = np.column_stack([cols[k] for k in ["ppa", "elo"] + list(groups)] + [cols["hfa"]])
        sd = X[base].std(0) if base.sum() > 10 else np.ones(X.shape[1])
        sd[sd == 0] = 1; sd[-1] = 1
        return X / sd

    def fit(X, mask):
        if mask.sum() < 30:
            w = np.zeros(X.shape[1]); w[0:3] = [0.3, 0.3, 1.0]; return w
        Xt, yt = X[mask], y[mask]
        f = lambda w: np.mean(np.log1p(np.exp(-(Xt @ w) * (2 * yt - 1)))) + L2 * np.sum(w[:-1] ** 2)
        return minimize(f, np.zeros(X.shape[1]), method="BFGS").x

    def walk(groups, seasons):
        X = matrix(groups); P = np.full(n, np.nan)
        for s in seasons:
            m = (S == s) & base
            if m.any():
                P[m] = sig(X[m] @ fit(X, base & (S < s)))
        return P

    fbs = np.array([g["hcl"] == "fbs" and g["acl"] == "fbs" for g in G])

    def score(P, seasons, extra=None):
        m = np.isin(S, seasons) & ~np.isnan(P) & base
        if extra is not None:
            m &= extra
        if not m.any():
            return None
        p = np.clip(P[m], 1e-6, 1 - 1e-6)
        return dict(log_loss=round(float(-np.mean(y[m] * np.log(p) + (1 - y[m]) * np.log(1 - p))), 4),
                    picked_winner=round(float(np.mean((p > .5) == (y[m] == 1))), 3), games=int(m.sum()))

    test = [s for s in range(TEST_FROM, season + 1)]
    selected = json.load(open(FEATS)) if os.path.exists(FEATS) else None
    if selected is None or ev == "workflow_dispatch":
        seasons = list(DEV) + test
        report = dict(updated=now.strftime("%Y-%m-%d %H:%M UTC"), dev_seasons=list(DEV), test_seasons=test,
                      market=dict(dev=score(np.where(base, kp, np.nan), DEV), test=score(np.where(base, kp, np.nan), test)))
        P0 = walk([], seasons); d0 = score(P0, DEV)
        report["current_model"] = dict(dev=d0, test=score(P0, test))
        report["each_group_added"] = {}
        for gname in GROUPS:
            P = walk([gname], seasons)
            report["each_group_added"][gname] = dict(dev=score(P, DEV), test=score(P, test))
        sel, best = [], d0["log_loss"] if d0 else 9
        pool = list(GROUPS)
        while pool:
            trials = [(score(walk(sel + [g], DEV), DEV)["log_loss"], g) for g in pool]
            l, g = min(trials)
            if l < best - 0.0005:
                sel.append(g); pool.remove(g); best = l
            else:
                break
        Ps = walk(sel, seasons)
        report["selected_groups"] = sel
        report["selected_model"] = dict(dev=score(Ps, DEV), test=score(Ps, test))
        Pa = walk(GROUPS, seasons)
        report["all_groups"] = dict(dev=score(Pa, DEV), test=score(Pa, test))
        mk = np.where(base, kp, np.nan)
        report["fbs_vs_fbs_only"] = dict(market=dict(dev=score(mk, DEV, fbs), test=score(mk, test, fbs)),
                                         selected_model=dict(dev=score(Ps, DEV, fbs), test=score(Ps, test, fbs)))
        report["download_problems"] = failed
        json.dump(report, open(EVAL, "w"), indent=1)
        json.dump(sel, open(FEATS, "w"))
        selected = sel
        print(json.dumps(report, indent=1))

    X = matrix(selected)
    p_past = sig(X @ fit(X, base & (S < season)))
    p_now = sig(X @ fit(X, base))
    games, past = [], []
    for i, g in enumerate(G):
        if g["season"] != season or np.isnan(kp[i]):
            continue
        sp = None if g["spread"] is None else -g["spread"]
        o = dict(id="cfb" + g["id"], d=g["d"], t=g["t"], wk=g["week"], a=g["a"], h=g["h"], ac=g["ac"], hc=g["hc"],
                 am=None if np.isnan(am[i]) else int(am[i]), hm=None if np.isnan(hm[i]) else int(hm[i]),
                 sp=sp, aso=-110 if sp is not None else None, hso=-110 if sp is not None else None,
                 tot=g["tot"], ov=-110 if g["tot"] is not None else None, un=-110 if g["tot"] is not None else None,
                 aq="", hq="", kp=round(float(kp[i]), 3), mp=round(float(p_now[i]), 3), ns=bool(g["neutral"]),
                 fbs=bool(fbs[i]))
        if g["done"]:
            o["sa"], o["sh"] = int(g["as_"]), int(g["hs"])
            if hasml[i]:
                past.append([season, g["week"], "REG" if g["stype"] == "regular" else "POST", g["d"], g["a"], g["h"],
                             int(g["as_"]), int(g["hs"]), int(am[i]), int(hm[i]), round(float(kp[i]), 3), round(float(p_past[i]), 3),
                             bool(fbs[i])])
        games.append(o)

    decided = [r for r in past if r[6] != r[7]]
    val = [(r, s_, o) for r in past for s_, p, o in (("h", r[11], r[9]), ("a", 1 - r[11], r[8])) if p * dec(o) - 1 >= .03]
    stats = dict(n=len(decided), book=sum((r[10] >= .5) == (r[7] > r[6]) for r in decided),
                 model=sum((r[11] >= .5) == (r[7] > r[6]) for r in decided),
                 vn=len(val), vw=sum(r[6] != r[7] and ((r[7] > r[6]) == (s_ == "h")) for r, s_, o in val),
                 vdog=sum(o > 0 for _, _, o in val),
                 vprofit=round(sum(0 if r[6] == r[7] else (100 * (dec(o) - 1) if (r[7] > r[6]) == (s_ == "h") else -100)
                                   for r, s_, o in val), 2), features=selected)
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
    json.dump(dict(updated=now.strftime("%Y-%m-%d %H:%M UTC"), season=season, games=games, past=past,
                   teams="", hist="", stats=stats), open(OUT, "w"), separators=(",", ":"))
    print("wrote", OUT, "| games", len(games), "| finished with moneylines", len(past), "| features", selected)


if __name__ == "__main__":
    run()
