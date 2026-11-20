"""
PAPER TRADER : gamma exposure (GEX) + flux d'ordres (delta), BTC perpetuel, donnees gratuites Deribit.
AUCUN ORDRE REEL : positions virtuelles, journal CSV. Un appel de step() = un cycle (toutes les 15 min).

v2 (2026-11-20) : la v1 n'a donne AUCUN signal en 205 cycles (regime NEG 100 % du temps, murs a 4 % / 12 % du spot,
cassure exigee). Murs limites a +-3 % du spot, cassure NEG anticipee (tolerance 3 %, flux >= 0.10).
Le compteur des 100 trades repart de zero avec la v2 (la v1 n'a produit aucun trade : rien n'est contamine).
================== REGLES FIGEES AVANT LE TEST (ne rien changer avant d'avoir 100 trades) ==================
Contexte (GEX calcule sur toutes les options BTC <= 60 jours, convention : calls +, puts -) :
  regime POS = GEX net au spot > 0 (les teneurs de marche freinent les mouvements)
  regime NEG = GEX net au spot < 0 (ils amplifient les mouvements)
  call_wall  = strike AU-DESSUS du spot avec le plus grand GEX de calls
  put_wall   = strike EN-DESSOUS du spot avec le plus grand GEX de puts (en valeur absolue)
Flux : delta_ratio = (volume acheteur agressif - vendeur agressif) / volume total, sur les 30 dernieres minutes (perp).
Signaux (verifies a chaque cycle, une seule position a la fois) :
  POS (rebond sur les murs) : LONG  si spot <= put_wall  * 1.004 et delta_ratio >= +0.15
                              SHORT si spot >= call_wall * 0.996 et delta_ratio <= -0.15
  NEG (cassure des murs)    : SHORT si spot <  put_wall  * 1.03 et delta_ratio <= -0.10
                              LONG  si spot >  call_wall * 0.97 et delta_ratio >= +0.10
  (murs = strike de plus grand GEX dans une bande de +-3 % autour du spot)
Sorties : stop 0.6 %, objectif 1.2 %, sortie temps 4 h. Stop avant objectif si les deux sont touches dans la meme minute.
Couts : 5 bps de frais + 2 bps de glissement par execution (donc ~14 bps l'aller-retour).
Verdict : a 100 trades clotures, moyenne nette > 0, p < 0.05, borne basse de l'IC > 0. A 60 trades : si t < -1, on arrete.
Les seuils (0.004, 0.15, 0.6 %, 1.2 %, 4 h) sont des choix a priori, PAS optimises. Ne pas les ajuster apres coup.
============================================================================================================
"""
import csv
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import requests

# ------------------------------------------------------------------ parametres figes
INSTR = "BTC-PERPETUAL"
CURRENCY = "BTC"
MAX_DAYS = 60
WALL_TOL = 0.004
FLOW_MIN = 0.15
WALL_RANGE = 0.03                # murs cherches dans +-3 % du spot (v1 : toute la chaine, murs trop loin)
NEG_TOL = 0.03                   # regime NEG : entree des que le spot est a moins de 3 % du mur, cote cassure
NEG_FLOW = 0.10
FLOW_WINDOW_MIN = 30
STOP, TARGET = 0.006, 0.012
MAX_HOLD_MIN = 240
FEE, SLIP = 5e-4, 2e-4
DATA = os.environ.get("PAPER_DATA", "data")     # sur Colab : un dossier Google Drive, sinon le journal disparait
LOG, TRADES, STATE = f"{DATA}/paper_log_v2.csv", f"{DATA}/paper_trades.csv", f"{DATA}/paper_state.json"   # v2 : nouvelle colonne 'bloque'
ND = NormalDist()
API = "https://www.deribit.com/api/v2/public"
MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


# ------------------------------------------------------------------ client Deribit (public, sans cle)
class DeribitClient:
    def _get(self, path, **params):
        for k in range(3):
            try:
                r = requests.get(f"{API}/{path}", params=params, timeout=30)
                r.raise_for_status()
                return r.json()["result"]
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"Deribit injoignable : {path}")

    def index_price(self):
        return float(self._get("get_index_price", index_name="btc_usd")["index_price"])

    def option_summaries(self):
        return self._get("get_book_summary_by_currency", currency=CURRENCY, kind="option")

    def trades(self, start_ms, end_ms):
        out, seen, cur = [], set(), start_ms
        for _ in range(40):
            res = self._get("get_last_trades_by_instrument_and_time", instrument_name=INSTR,
                            start_timestamp=cur, end_timestamp=end_ms, count=1000, sorting="asc")
            new = [t for t in res["trades"] if t["trade_id"] not in seen]
            for t in new:
                seen.add(t["trade_id"])
            out += new
            if not res.get("has_more") or not res["trades"]:
                break
            cur = res["trades"][-1]["timestamp"]
        return out

    def candles(self, start_ms, end_ms):
        r = self._get("get_tradingview_chart_data", instrument_name=INSTR,
                      start_timestamp=start_ms, end_timestamp=end_ms, resolution="1")
        if r.get("status") != "ok":
            return []
        return [dict(t=r["ticks"][i], o=r["open"][i], h=r["high"][i], l=r["low"][i], c=r["close"][i])
                for i in range(len(r["ticks"]))]


# ------------------------------------------------------------------ GEX
def parse_option(name):
    m = re.match(r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$", name)
    if not m:
        return None
    d, mon, y, k, cp = m.groups()
    exp = datetime(2000 + int(y), MONTHS[mon], int(d), 8, 0, tzinfo=timezone.utc)   # echeance 08:00 UTC
    return exp.timestamp(), float(k.replace("d", ".")), 1.0 if cp == "C" else -1.0


def _gamma(S, K, T, iv):
    s = iv * np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * s * s) / s
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * s)


def gex_levels(summaries, spot, now_ts):
    rows = []
    for o in summaries:
        p = parse_option(o["instrument_name"])
        oi, iv = o.get("open_interest") or 0, o.get("mark_iv") or 0
        if not p or oi <= 0 or iv <= 0:
            continue
        T = (p[0] - now_ts) / (365.25 * 86400)
        if T <= 0 or T > MAX_DAYS / 365.25:
            continue
        rows.append((p[1], max(T, 1 / (365.25 * 24)), iv, oi, p[2]))
    if len(rows) < 30:
        return None
    K, T, iv, oi, sg = (np.array(c, dtype=float) for c in zip(*rows))
    if np.median(iv) > 5:           # volatilite en pourcents
        iv = iv / 100.0
    iv = np.clip(iv, 0.05, 5.0)

    def gex_at(S):                  # GEX en $ par mouvement de 1 %
        return sg * oi * _gamma(S, K, T, iv) * S * S * 0.01

    per = gex_at(spot)
    net = float(per.sum())
    strikes = np.unique(K)
    call_by = {k: per[(K == k) & (sg > 0)].sum() for k in strikes}
    put_by = {k: -per[(K == k) & (sg < 0)].sum() for k in strikes}
    above = [k for k in strikes if spot < k <= spot * (1 + WALL_RANGE)]
    below = [k for k in strikes if spot * (1 - WALL_RANGE) <= k < spot]
    call_wall = max(above, key=lambda k: call_by[k]) if above and max(call_by[k] for k in above) > 0 else None
    put_wall = max(below, key=lambda k: put_by[k]) if below and max(put_by[k] for k in below) > 0 else None
    grid = spot * (1 + np.linspace(-0.15, 0.15, 61))
    tot = np.array([gex_at(s).sum() for s in grid])
    flip = None
    for i in range(len(grid) - 1):
        if tot[i] * tot[i + 1] < 0:
            c = grid[i] - tot[i] * (grid[i + 1] - grid[i]) / (tot[i + 1] - tot[i])
            if flip is None or abs(c - spot) < abs(flip - spot):
                flip = float(c)
    return dict(net_gex=net, regime="POS" if net > 0 else "NEG", call_wall=call_wall, put_wall=put_wall, flip=flip)


# ------------------------------------------------------------------ flux d'ordres
def delta_ratio(trades):
    buy = sum(t["amount"] for t in trades if t["direction"] == "buy")
    sell = sum(t["amount"] for t in trades if t["direction"] == "sell")
    tot = buy + sell
    return (buy - sell) / tot if tot > 0 else 0.0


# ------------------------------------------------------------------ decision
def decide(spot, g, dr):
    """Retourne +1 (long), -1 (short) ou 0. Regles figees (voir en-tete)."""
    if g is None:
        return 0, ""
    cw, pw = g["call_wall"], g["put_wall"]
    if g["regime"] == "POS":
        if pw and spot <= pw * (1 + WALL_TOL) and dr >= FLOW_MIN:
            return 1, "POS rebond mur put"
        if cw and spot >= cw * (1 - WALL_TOL) and dr <= -FLOW_MIN:
            return -1, "POS rejet mur call"
    else:
        if pw and spot < pw * (1 + NEG_TOL) and dr <= -NEG_FLOW:
            return -1, "NEG cassure mur put"
        if cw and spot > cw * (1 - NEG_TOL) and dr >= NEG_FLOW:
            return 1, "NEG cassure mur call"
    return 0, ""


def blocker(spot, g, dr):
    """Pourquoi aucun signal ce cycle (colonne 'bloque' du journal)."""
    if g is None:
        return "pas_de_gex"
    cw, pw = g["call_wall"], g["put_wall"]
    if g["regime"] == "POS":
        near, flow = (pw and spot <= pw * (1 + WALL_TOL)) or (cw and spot >= cw * (1 - WALL_TOL)), FLOW_MIN
    else:
        near, flow = (pw and spot < pw * (1 + NEG_TOL)) or (cw and spot > cw * (1 - NEG_TOL)), NEG_FLOW
    if not pw and not cw:
        return "aucun_mur"
    if not near:
        return "loin_des_murs"
    return "flux_insuffisant" if abs(dr) < flow else "flux_sens_oppose"


# ------------------------------------------------------------------ positions virtuelles
def open_position(side, spot, now_ts, why, g):
    entry = spot * (1 + side * SLIP)
    return dict(side=side, entry=entry, t0=now_ts, why=why, regime=g["regime"],
                stop=entry * (1 - side * STOP), target=entry * (1 + side * TARGET), last_ms=int(now_ts * 1000))


def manage(pos, candles, now_ts):
    """Parcourt les bougies d'une minute depuis la derniere verification. Retourne (trade ou None, pos)."""
    s = pos["side"]
    for c in candles:
        if c["t"] <= pos["last_ms"]:
            continue
        pos["last_ms"] = c["t"]
        hit_stop = c["l"] <= pos["stop"] if s > 0 else c["h"] >= pos["stop"]
        hit_tp = c["h"] >= pos["target"] if s > 0 else c["l"] <= pos["target"]
        if hit_stop:
            return close(pos, pos["stop"], c["t"] / 1000, "stop", market=True), None
        if hit_tp:
            return close(pos, pos["target"], c["t"] / 1000, "objectif", market=False), None
        if c["t"] / 1000 - pos["t0"] >= MAX_HOLD_MIN * 60:
            return close(pos, c["c"], c["t"] / 1000, "temps", market=True), None
    return None, pos


def close(pos, px, ts, reason, market):
    s = pos["side"]
    exit_px = px * (1 - s * SLIP) if market else px
    gross = s * (exit_px / pos["entry"] - 1)
    net = gross - 2 * FEE
    return dict(t_entry=iso(pos["t0"]), t_exit=iso(ts), side="LONG" if s > 0 else "SHORT", entry=round(pos["entry"], 2),
                exit=round(exit_px, 2), reason=reason, regime=pos["regime"], why=pos["why"],
                gross_pct=round(gross * 100, 4), net_pct=round(net * 100, 4))


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ------------------------------------------------------------------ cycle
def _append(path, row):
    os.makedirs(DATA, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def load_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {"pos": None}


def save_state(st):
    os.makedirs(DATA, exist_ok=True)
    json.dump(st, open(STATE, "w"))


def step(client=None, now_ts=None):
    client = client or DeribitClient()
    now_ts = now_ts or time.time()
    st = load_state()
    msg = []
    # 1) gerer la position ouverte avec les bougies 1 min depuis la derniere verification
    if st["pos"]:
        cs = client.candles(int(st["pos"]["last_ms"]), int(now_ts * 1000))
        tr, st["pos"] = manage(st["pos"], cs, now_ts)
        if tr:
            _append(TRADES, tr)
            msg.append(f"CLOTURE {tr['side']} {tr['reason']} net {tr['net_pct']:+.2f} %")
    # 2) contexte GEX + flux
    spot = client.index_price()
    g = gex_levels(client.option_summaries(), spot, now_ts)
    tr_list = client.trades(int((now_ts - FLOW_WINDOW_MIN * 60) * 1000), int(now_ts * 1000))
    dr = delta_ratio(tr_list)
    # 3) decision (seulement si aucune position)
    side, why = (0, "")
    if st["pos"] is None:
        side, why = decide(spot, g, dr)
        if side and g is not None:
            st["pos"] = open_position(side, spot, now_ts, why, g)
            msg.append(f"OUVERTURE {'LONG' if side > 0 else 'SHORT'} ({why}) @ {spot:.0f}")
    _append(LOG, dict(time=iso(now_ts), spot=round(spot, 2), regime=(g or {}).get("regime", ""),
                      net_gex=round((g or {}).get("net_gex", 0)), flip=round((g or {}).get("flip") or 0),
                      call_wall=(g or {}).get("call_wall") or "", put_wall=(g or {}).get("put_wall") or "",
                      delta_ratio=round(dr, 3), n_trades=len(tr_list), signal=side, bloque="" if side else ("position_ouverte" if st["pos"] else blocker(spot, g, dr)),
                      position=0 if not st["pos"] else st["pos"]["side"]))
    save_state(st)
    return msg


# ------------------------------------------------------------------ evaluation (a lancer quand on veut)
def evaluate(path=TRADES, seed=7):
    import pandas as pd
    if not os.path.exists(path):
        print("Aucun trade cloture pour l'instant.")
        return
    d = pd.read_csv(path)
    x = d["net_pct"].values / 100
    n = len(x)
    print(f"Trades clotures : {n}")
    if n < 5:
        print("Trop tot : aucune conclusion possible.")
        return
    m, sd = x.mean(), x.std(ddof=1)
    t = m / (sd / math.sqrt(n)) if sd > 0 else 0.0
    p = 2 * (1 - ND.cdf(abs(t)))
    rng = np.random.default_rng(seed)
    bs = np.array([rng.choice(x, n).mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"Moyenne nette par trade : {m * 100:+.3f} %   (brute {d['gross_pct'].mean():+.3f} %)")
    print(f"Taux de reussite : {(x > 0).mean() * 100:.0f} %   t = {t:.2f}   p = {p:.3f}   IC 95 % [{lo * 100:+.3f} ; {hi * 100:+.3f}] %")
    print("Par regime :", {k: (len(v), round(v['net_pct'].mean(), 3)) for k, v in d.groupby("regime")})
    print("Par sortie :", {k: len(v) for k, v in d.groupby("reason")})
    if n >= 100:
        ok = m > 0 and p < 0.05 and lo > 0
        print("VERDICT (100 trades) :", "EDGE PROBABLE : a confirmer sur une 2e periode" if ok else "NON DEMONTRE : on n'a pas d'edge prouve")
    elif n >= 60 and t < -1:
        print("REGLE D'ARRET : t < -1 a 60 trades -> on abandonne cette strategie.")
    else:
        print(f"Pas de verdict avant 100 trades (reste {100 - n}).")


def loop(every_min=15):
    """Pour Colab : tourne tant que l'onglet reste ouvert."""
    while True:
        try:
            for m in step():
                print(iso(time.time()), m)
        except Exception as e:
            print("erreur cycle :", e)
        time.sleep(every_min * 60 - (time.time() % (every_min * 60)) + 5)


if __name__ == "__main__":
    for m in step():
        print(m)
