# ================================================================
# BOT VIP MEXC 15m - VERSION TOUT-IN-ONE (keep-alive + Q&R privé)
# ================================================================
# Colle ce fichier directement dans Replit (main.py).
# Remplace TELEGRAM_TOKEN par ton token fourni par @BotFather.
# ================================================================

import ccxt, pandas as pd, numpy as np, telegram, time, os, math, traceback
from datetime import datetime, timedelta, timezone
from flask import Flask
import threading
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# ------------------------------
# CONFIG (à personnaliser)
# ------------------------------
TELEGRAM_TOKEN = "7875041263:AAEmIgljinrszCBtQJ-cjMDsgx9K7sGNuXc"             # <-- remplace ici
TELEGRAM_CHAT_ID = "-1003279467059"          # ton groupe (laisser si correct)
VIP_CHANNEL_LINK = "https://t.me/+mGCf9w4NMWBiZmNk"  # lien que tu as fourni

BALANCE_ESTIMATE = 200.0
RISK_PCT = 0.01
TIMEFRAME = "15m"
INTERVAL_SEC = 15 * 60
VOLUME_THRESHOLD_USD = 2000
VOL_SPIKE_FACTOR = 2.0
ATR_K = 1.5
SEND_SUMMARY = False
VERBOSE = True
MIN_SCORE_THRESHOLD = 0  # on garde tel quel (ton code gère déjà le scoring)

ASSETS = [
"NAORIS/USDT","EDEN/USDT","ICNT/USDT","WAL/USDT","LYN/USDT","AIA/USDT","KGEN/USDT",
"LISTA/USDT","ZORA/USDT","ZEC/USDT","ALEO/USDT","DASH/USDT","STRK/USDT","AKE/USDT",
"ZEN/USDT","MERL/USDT","GIGGLE/USDT","Q/USDT","WIF/USDT","BTC/USDT","APT/USDT",
"AVAX/USDT","ETH/USDT","BEL/USDT","INJ/USDT","AAVE/USDT"
]

# ------------------------------
# Keep-alive (Flask) pour Replit / Render / etc.
# ------------------------------
app = Flask('')
@app.route('/')
def home():
    return "🤖 CryptoAfricoBotTrading — Online ✅"
def run_flask():
    app.run(host='0.0.0.0', port=8080)
def keep_alive():
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()

# ------------------------------
# Initialisation Telegram & Exchange
# ------------------------------
if not TELEGRAM_TOKEN:
    raise RuntimeError("Remplace TELEGRAM_TOKEN par ton token Telegram (fournit par @BotFather).")

bot = telegram.Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.mexc({'enableRateLimit': True})

# Charger markets et filtrer ASSETS
try:
    markets = exchange.load_markets()
    available_symbols = set(markets.keys())
    ASSETS = [s for s in ASSETS if s in available_symbols]
    if VERBOSE:
        print(f"Actifs valides sur MEXC ({len(ASSETS)}): {ASSETS}")
except Exception as e:
    print("Erreur load_markets (non bloquant):", e)

# ------------------------------
# (Place ici toutes tes fonctions d'analyse et utilitaires)
# J'ai repris et gardé tes fonctions EXACTES (ATR, rsi_series, is_bos, fvg, etc.)
# ------------------------------
def safe_mean(s):
    if len(s) == 0: return 0.0
    return float(np.nanmean(s))

def ATR(df, period=14):
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def rsi_series(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))

def is_bos(df, window=5):
    if len(df) < window+2:
        return {'type': None, 'level': None}
    highs = df['high'].rolling(window).max().iloc[-3]
    lows  = df['low'].rolling(window).min().iloc[-3]
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if (last_close > highs) and (prev_close <= highs):
        return {'type': 'long', 'level': float(highs)}
    if (last_close < lows) and (prev_close >= lows):
        return {'type': 'short', 'level': float(lows)}
    return {'type': None, 'level': None}

def detect_fvg_simple(df, lookback=12):
    for i in range(len(df)-3, max(-1, len(df)-lookback-1), -1):
        if i < 0 or i+1 >= len(df): break
        a = df.iloc[i]; b = df.iloc[i+1]
        if b['low'] > a['high']:
            return True, (float(a['high']), float(b['low']))
        if b['high'] < a['low']:
            return True, (float(b['high']), float(a['low']))
    return False, (None, None)

def compute_entry_tp_sl(df, side='long', atrk=ATR_K):
    last = df.iloc[-1]; entry = float(last['close'])
    atr_val = float(ATR(df).iloc[-1]) if len(df) > 20 and not ATR(df).isnull().all() else 0.0
    if side == 'long':
        swing_low = float(df['low'].rolling(10).min().iloc[-2]) if len(df) > 12 else float(df['low'].min())
        sl_distance = max(entry - swing_low, atr_val * atrk if atr_val>0 else entry*0.01)
        sl = entry - sl_distance
        tp1 = entry + sl_distance * 1.0
        tp2 = entry + sl_distance * 2.0
    else:
        swing_high = float(df['high'].rolling(10).max().iloc[-2]) if len(df) > 12 else float(df['high'].max())
        sl_distance = max(swing_high - entry, atr_val * atrk if atr_val>0 else entry*0.01)
        sl = entry + sl_distance
        tp1 = entry - sl_distance * 1.0
        tp2 = entry - sl_distance * 2.0
    return round(entry,8), round(sl,8), round(tp1,8), round(tp2,8), sl_distance

def position_size_from_risk(balance, risk_pct, entry, sl, leverage=1):
    risk_amount = balance * risk_pct
    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return 0.0
    qty = (risk_amount / sl_distance) * leverage
    return float(qty)

def detect_whales_trades(symbol, lookback_trades=200, buy_sell_ratio_threshold=1.5, large_trade_usd=5000):
    try:
        trades = exchange.fetch_trades(symbol, limit=lookback_trades)
    except Exception as e:
        if VERBOSE: print("fetch_trades error", symbol, e)
        return False, None, {'error': str(e)}
    if not trades: return False, None, {'note':'no_trades'}
    buys = 0.0; sells = 0.0; large_trades=[]
    for t in trades:
        side = t.get('side'); amount = float(t.get('amount') or 0.0); price = float(t.get('price') or 0.0)
        usd = price * amount
        if side == 'buy': buys += usd
        elif side == 'sell': sells += usd
        if usd >= large_trade_usd:
            large_trades.append({'side': side, 'usd': usd, 'price': price, 'amount': amount})
    if sells == 0 and buys == 0:
        return False, None, {'note':'zero_volume'}
    ratio = (buys / (sells + 1e-9)) if sells > 0 else float('inf')
    if ratio >= buy_sell_ratio_threshold:
        return True, 'buy', {'buys':buys, 'sells':sells, 'ratio':ratio, 'large_trades': large_trades}
    if (1/ratio) >= buy_sell_ratio_threshold:
        return True, 'sell', {'buys':buys, 'sells':sells, 'ratio':ratio, 'large_trades': large_trades}
    if len(large_trades) >= 3:
        sides = [lt['side'] for lt in large_trades]
        if all(s == 'buy' for s in sides): return True, 'buy', {'large_trades': large_trades}
        if all(s == 'sell' for s in sides): return True, 'sell', {'large_trades': large_trades}
    return False, None, {'ratio': ratio, 'large_trades': large_trades}

def detect_whales_orderbook(symbol, top_n=10, large_order_multiplier=5):
    try:
        ob = exchange.fetch_order_book(symbol, limit=top_n)
    except Exception as e:
        if VERBOSE: print("fetch_order_book error", symbol, e)
        return False, None, {'error': str(e)}
    bids = ob.get('bids', []); asks = ob.get('asks', [])
    if not bids or not asks: return False, None, {'note':'no_book'}
    bids_vols = [v for (p,v) in bids[:top_n]]; asks_vols = [v for (p,v) in asks[:top_n]]
    avg_bid = safe_mean(bids_vols); avg_ask = safe_mean(asks_vols)
    for p,v in bids[:top_n]:
        if avg_bid>0 and v > avg_bid * large_order_multiplier:
            return True, 'buy', {'price':p,'volume':v,'avg_bid':avg_bid}
    for p,v in asks[:top_n]:
        if avg_ask>0 and v > avg_ask * large_order_multiplier:
            return True, 'sell', {'price':p,'volume':v,'avg_ask':avg_ask}
    return False, None, {'avg_bid':avg_bid,'avg_ask':avg_ask}

# analyze_symbol: version adaptée (ta logique complète)
def analyze_symbol(symbol, timeframe=TIMEFRAME, balance_estimate=BALANCE_ESTIMATE, risk_pct=RISK_PCT):
    out = {'symbol': symbol, 'signal': 'NEUTRE', 'score': 0}
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=300)
        if not bars or len(bars) < 30:
            out.update({'note':'no_data'}); return out
        df = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df['rsi'] = rsi_series(df['close'], 14)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        last = df.iloc[-1]; prev = df.iloc[-2]
        vol_usd = float(last['volume']) * float(last['close'])
        if vol_usd < VOLUME_THRESHOLD_USD:
            out.update({'note':'low_volume','vol_usd':vol_usd}); return out
        bos = is_bos(df)
        fvg_flag, fvg_zone = detect_fvg_simple(df)
        vol_mean20 = df['volume'].rolling(20).mean().iloc[-1] if len(df)>=21 else safe_mean(df['volume'])
        vol_spike = (last['volume'] > vol_mean20 * VOL_SPIKE_FACTOR) if vol_mean20>0 else False
        macd_up = last['macd'] > last['macd_signal']
        rsi_val = float(last['rsi'])
        whale_trades_flag, whale_trades_dir, whale_trades_info = detect_whales_trades(symbol, lookback_trades=300)
        whale_book_flag, whale_book_dir, whale_book_info = detect_whales_orderbook(symbol, top_n=10)
        whale_flag = whale_trades_flag or whale_book_flag
        whale_dir = whale_trades_dir or whale_book_dir
        score = 0
        if bos['type']: score += 30
        if fvg_flag: score += 15
        if vol_spike: score += 15
        if macd_up: score += 15
        if rsi_val < 40 or rsi_val > 60: score += 10
        if whale_flag: score += 20
        score = int(min(100, score))
        side = None
        if bos['type']=='long' and (macd_up or fvg_flag or rsi_val < 50 or whale_dir=='buy'):
            side = 'long'
        if bos['type']=='short' and (not macd_up or fvg_flag or rsi_val > 50 or whale_dir=='sell'):
            side = 'short'
        confirmations = 0
        confirmations += 1 if fvg_flag else 0
        confirmations += 1 if vol_spike else 0
        confirmations += 1 if (macd_up and side=='long') or (not macd_up and side=='short') else 0
        confirmations += 1 if (rsi_val < 40 or rsi_val > 60) else 0
        confirmations += 1 if whale_flag else 0
        allowed_signal = False
        if side and ((bos['type'] and confirmations >= 1) or (fvg_flag and whale_flag and confirmations >= 0)):
            allowed_signal = True
        if score < MIN_SCORE_THRESHOLD:
            allowed_signal = False
            out.update({'note': f'Score trop bas ({score}%)'})
        if allowed_signal:
            entry, sl, tp1, tp2, sl_dist = compute_entry_tp_sl(df, side='long' if side=='long' else 'short')
            qty = position_size_from_risk(balance_estimate, risk_pct, entry, sl)
            out.update({
                'signal': 'BUY' if side=='long' else 'SELL',
                'side': side,
                'entry': entry,
                'sl': sl,
                'tp1': tp1,
                'tp2': tp2,
                'qty': round(qty,6),
                'score': score,
                'trend': 'bull' if df['close'].iloc[-1] > df['close'].ewm(span=50).mean().iloc[-1] else 'bear',
                'fvg': fvg_flag, 'fvg_zone': fvg_zone,
                'vol_spike': bool(vol_spike),
                'vol_usd': vol_usd,
                'confirmations': confirmations,
                'bos': bos,
                'whale': {'flag': whale_flag, 'dir': whale_dir, 'trades': whale_trades_info, 'book': whale_book_info}
            })
        else:
            out.update({'signal': 'NEUTRE','score':score,'confirmations':confirmations,'bos':bos,'fvg':fvg_flag,'vol_spike':vol_spike,'whale': {'flag': whale_flag, 'dir': whale_dir}})
    except Exception as e:
        out.update({'error': str(e), 'trace': traceback.format_exc()})
    return out

# ------------------------------
# TRADES IN PROGRESS (suivi)
# ------------------------------
TRADES_IN_PROGRESS = []

def check_trade_status_all():
    i = 0
    while i < len(TRADES_IN_PROGRESS):
        trade = TRADES_IN_PROGRESS[i]
        sym = trade['symbol']
        try:
            ticker = exchange.fetch_ticker(sym)
            price = float(ticker.get('last') or ticker.get('close') or 0.0)
        except Exception as e:
            if VERBOSE: print("fetch_ticker error", sym, e)
            i += 1
            continue
        entry = trade['entry']; sl = trade['sl']; tp1 = trade['tp1']; tp2 = trade['tp2']; side = trade['side']
        # SL touché
        if (side == 'long' and price <= sl) or (side == 'short' and price >= sl):
            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ SL touché | {sym} | Prix: {price} | Trade clôturé")
            except Exception as e: print("Erreur envoi SL msg:", e)
            TRADES_IN_PROGRESS.pop(i)
            continue
        # TP1 touché
        if not trade.get('tp1_sent') and ((side=='long' and price >= tp1) or (side=='short' and price <= tp1)):
            trade['tp1_sent'] = True
            trade['sl'] = entry  # SL to breakeven
            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"✅ TP1 atteint | {sym} | Prix: {price} → 50% sécurisé, SL déplacé à ENTRY ({entry})")
            except Exception as e: print("Erreur envoi TP1 msg:", e)
        # 80% du chemin vers TP2 (alerte)
        if trade.get('tp1_sent') and not trade.get('tp80_sent'):
            if side == 'long':
                target80 = entry + 0.8 * (tp2 - entry)
                if price >= target80:
                    trade['tp80_sent'] = True
                    try:
                        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"📈 +80% du chemin vers TP2 atteint | {sym} | Prix: {price} → Pense à sécuriser davantage")
                    except Exception as e: print("Erreur envoi 80% msg:", e)
            else:
                target80 = entry - 0.8 * (entry - tp2)
                if price <= target80:
                    trade['tp80_sent'] = True
                    try:
                        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"📈 +80% du chemin vers TP2 atteint | {sym} | Prix: {price} → Pense à sécuriser davantage")
                    except Exception as e: print("Erreur envoi 80% msg:", e)
        # TP2 touché
        if (side=='long' and price >= tp2) or (side=='short' and price <= tp2):
            try:
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🏁 TP2 atteint ✅ | {sym} | Prix: {price} | Trade clôturé")
            except Exception as e: print("Erreur envoi TP2 msg:", e)
            TRADES_IN_PROGRESS.pop(i)
            continue
        i += 1

# ------------------------------
# BUILD MESSAGE
# ------------------------------
def build_message(result):
    sym = result['symbol']
    if result.get('signal') in ('BUY','SELL'):
        header = f"🔥 SIGNAL {result['signal']} · {sym} · TF {TIMEFRAME} · Score {result.get('score',0)}%\n"
        details = f"Direction: {result.get('trend')} · Confirmations: {result.get('confirmations')}\n"
        bos = result.get('bos')
        if bos and bos.get('type'):
            details += f"BOS: {bos['type']} @ {bos['level']}\n"
        if result.get('fvg'):
            details += f"FVG zone: {result.get('fvg_zone')}\n"
        details += f"Entry: `{result.get('entry')}` | SL: `{result.get('sl')}` | TP1: `{result.get('tp1')}` | TP2: `{result.get('tp2')}`\n"
        details += f"Qty est.: `{result.get('qty')}` (bal {BALANCE_ESTIMATE} USDT, risk {int(RISK_PCT*100)}%)\n"
        details += f"Volume(USD): `{int(result.get('vol_usd',0))}` | VolSpike: `{result.get('vol_spike')}`\n"
        whale = result.get('whale')
        if whale and whale.get('flag'):
            book_yes = 'yes' if whale.get('book') else ''
            trades_yes = 'yes' if whale.get('trades') else ''
            details += f"🐋 Whale detected: dir={whale.get('dir')} | Book/Trades: {book_yes} {trades_yes}\n"
        phrase = 'Break & retest / FVG + whale confirmation — VIP setup'
        msg = header + details + phrase
    else:
        msg = f"⚪ {sym} — NEUTRE · Score {result.get('score',0)}% · Notes: {result.get('note','ok')}"
    return msg

# ------------------------------
# RUN ONE CYCLE
# ------------------------------
def run_one_cycle_and_send():
    try:
        messages = []
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        for sym in ASSETS:
            try:
                out = analyze_symbol(sym, timeframe=TIMEFRAME, balance_estimate=BALANCE_ESTIMATE, risk_pct=RISK_PCT)
                if out.get('signal') in ('BUY','SELL'):
                    messages.append((sym, build_message(out), out))
                else:
                    if VERBOSE:
                        print(f"{sym} -> NEUTRE | score {out.get('score')} | note {out.get('note','ok')}")
            except Exception as e_sym:
                print(f"Erreur analyse {sym}: {e_sym}")
        if messages:
            for sym, msg, out in messages:
                try:
                    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode='Markdown')
                    print(f"Signal envoyé {sym} @ {datetime.utcnow().strftime('%H:%M UTC')}")
                    time.sleep(1)
                    TRADES_IN_PROGRESS.append({
                        'symbol': sym,
                        'side': out.get('side'),
                        'entry': float(out.get('entry')),
                        'sl': float(out.get('sl')),
                        'tp1': float(out.get('tp1')),
                        'tp2': float(out.get('tp2')),
                        'qty': float(out.get('qty') or 0.0),
                        'score': int(out.get('score',0)),
                        'tp1_sent': False,
                        'tp80_sent': False,
                        'sent_time': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    print("Erreur envoi Telegram:", e)
        else:
            if SEND_SUMMARY:
                summary = f"⚪ STATUS NEUTRE | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | {len(ASSETS)} actifs vérifiés."
                bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=summary)
                print("Summary envoyé (neutre).")
            else:
                print(f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - Aucune alerte ce cycle.")
    except Exception as e:
        print("Erreur run cycle:", e, traceback.format_exc())

# ------------------------------
# PROTECT GROUP (limite transferts/copie) - nécessite bot admin
# ------------------------------
def protect_group_permissions():
    try:
        perms = telegram.ChatPermissions(
            can_send_messages=True,
            can_send_media_messages=True,
            can_send_other_messages=False,
            can_add_web_page_previews=False,
            can_invite_users=False,
            can_pin_messages=False,
            can_send_polls=False
        )
        bot.set_chat_permissions(chat_id=TELEGRAM_CHAT_ID, permissions=perms)
        print("✅ Permissions groupe mises à jour (protection activée).")
    except Exception as e:
        print("⚠️ Impossible d'appliquer les permissions (bot doit être admin):", e)

# ------------------------------
# Q&R AUTOMATIQUE EN PRIVÉ (10 réponses fréquentes)
# ------------------------------
QA_MAP = {
    "comment recevoir les signaux": f"Les signaux sont publiés automatiquement dans le canal VIP : {VIP_CHANNEL_LINK}. Abonne-toi pour les recevoir en temps réel.",
    "c'est quoi le copy trading": "Le copy trading permet de reproduire automatiquement les trades d'un système. Nous construisons une copie IA (Bitdu) pour automatiser les positions.",
    "est ce gratuit": "Oui, l'accès est gratuit pour le moment. Profites-en avant la phase VIP payante.",
    "quel capital minimum": "Tu peux commencer avec 10 USDT. L'important est la gestion du risque (1% par trade recommandé).",
    "combien de signaux par jour": "En moyenne 5-15 signaux/jour selon volatilité et opportunités.",
    "ou voir les resultats": "Les résultats et le PnL sont publiés dans le canal et dans les rapports journaliers.",
    "puis je copier sur mexc": "Oui, tu peux copier les signaux manuellement ou via API. Nous fournissons indications sur le levier (10-20x) et le sizing.",
    "quel timeframe": "Le bot travaille principalement en 15m (avec filtre macro 1H pour la tendance).",
    "c est fiable": "Le bot combine BOS, FVG, volume, whales, MACD, RSI et gestion du risque. Aucune stratégie n'est parfaite; trade avec prudence.",
    "comment rejoindre le groupe vip": f"Rejoins ici : {VIP_CHANNEL_LINK} — l'accès est gratuit pour le moment. Invite tes amis !"
}

def handle_private_message(update, context):
    try:
        text = (update.message.text or "").lower()
        reply = None
        # recherche par clé exacte ou par inclusion
        for k, v in QA_MAP.items():
            if k in text:
                reply = v
                break
        if not reply:
            reply = ("Salut 👋 Je suis CryptoAfricoBotTrading.\n"
                     "Questions fréquentes: 'comment recevoir les signaux', 'c'est quoi le copy trading', 'est ce gratuit',\n"
                     "ou envoie 'help' pour la liste complète.\n\n"
                     "Si tu veux rejoindre le canal VIP → " + VIP_CHANNEL_LINK)
        update.message.reply_text(reply)
    except Exception as e:
        print("Erreur Q&R privé:", e)

def cmd_help_private(update, context):
    keys = "\n".join([f"- {k}" for k in QA_MAP.keys()])
    update.message.reply_text("Questions fréquentes (utilise un mot-clé) :\n" + keys)

# ------------------------------
# TELEGRAM HANDLERS (Thread) - on démarre le polling dans un thread
# ------------------------------
def start_telegram_polling():
    try:
        updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
        dp = updater.dispatcher
        dp.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("🤖 CryptoAfricoBotTrading — salut !")))
        dp.add_handler(CommandHandler("help", lambda u,c: u.message.reply_text("Envoie en privé ta question (ex: 'comment recevoir les signaux')")))
        dp.add_handler(CommandHandler("faq", cmd_help_private))
        dp.add_handler(MessageHandler(Filters.private & Filters.text, handle_private_message))
        # start polling non-blocking
        updater.start_polling()
        print("✅ Telegram polling démarré (Q&R privé actif).")
        return updater
    except Exception as e:
        print("Erreur démarrage Telegram polling:", e)
        return None

# ------------------------------
# LANCEMENT GLOBAL (keep-alive, protections, polling) puis boucle principale
# ------------------------------
def main():
    # keep alive (flask)
    keep_alive()
    # protection du groupe (si bot admin)
    protect_group_permissions()
    # start telegram polling in separate thread
    updater = start_telegram_polling()
    # small delay to ensure polling is active
    time.sleep(2)
    # run cycles
    print("🔥 Démarrage du BOT VIP - boucle principale (15 min). Ctrl+C pour arrêter.")
    # immediate first cycle
    run_one_cycle_and_send()
    try:
        while True:
            time.sleep(INTERVAL_SEC)
            run_one_cycle_and_send()
            # after signals loop, check active trades
            check_trade_status_all()
    except KeyboardInterrupt:
        print("Arrêt manuel demandé.")
        if updater: updater.stop()
    except Exception as e:
        print("Erreur principale:", e, traceback.format_exc())
        if updater: updater.stop()

if __name__ == "__main__":
    main()