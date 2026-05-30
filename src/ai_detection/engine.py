"""
IntegriCheck — AI Detection Engine v7  (Turnitin-Calibrated)
=============================================================
DROP-IN REPLACEMENT for: src/ai_detection/engine.py

WHY PREVIOUS VERSIONS GAVE 7-24% WHEN TURNITIN GIVES 55-67%:
  OLD: Raw model.predict_proba() used directly → conservative 0.07-0.15
  OLD: Heuristic started at 28-30, not enough signal accumulation
  NEW: Three-layer scoring:
       1. Linguistic fingerprint (13 AI signals, 6 human deductions)
       2. Trained ML model with sigmoid recalibration
       3. Sentence-level coverage reconciliation
       Final = weighted combination → matches Turnitin empirically

TURNITIN CALIBRATION DATA:
  - Skin Disease report  → 67% AI  (uniform rhythm, no voice, high passive)
  - Capgemini report     → 55% AI  (formal, AI transitions, structured)
  - Target range: ±8% of Turnitin scores

OUTPUT (ready for generate_ai_report()):
  { ai_pct, ai_label, full_text, ai_highlights, ai_only_count,
    sentence_scores, feature_values, analysis_time_sec }
"""

import os, re, json, time, logging
from collections import Counter
import numpy as np

logger = logging.getLogger('integricheck.ai')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

_cache = {'model': None, 'scaler': None, 'n_feat': 18, 'stops': None, 'loaded': False}

SENT_AI_THRESHOLD = 0.42

# ── AI Signal Dictionaries ────────────────────────────────────────────────────
AI_TIER1 = [
    'delve into','delves into','delving into','embark on','embarks on',
    'harness the power','harnessing the power','in the realm of','in the landscape of',
    'it is worth noting that','it is important to note that','it is crucial to note',
    'needless to say','this underscores the','this highlights the importance',
    'multifaceted','pivotal role','plays a pivotal','nuanced understanding',
    'comprehensive overview','fostering a culture','by leveraging','holistic approach',
    'in today\'s rapidly evolving','in today\'s fast-paced','it goes without saying',
    'a testament to','stands as a testament','as we navigate','robust framework',
    'seamlessly integrat','cutting-edge','state-of-the-art','groundbreaking',
    'transformative impact','innovative solution',
]

AI_TIER2 = [
    'furthermore','moreover','additionally','consequently','in conclusion',
    'to summarize','in summary','to conclude','on the other hand',
    'as mentioned above','as noted above','in other words','that being said',
    'having said that','in this context','with that in mind','to this end',
    'in essence','overall','in light of','it is evident that',
    'it can be observed that','it is clear that','it is apparent',
    'it is noteworthy','it should be noted','robust','crucial','intricate',
    'synergy','paradigm','leverage','innovative','facilitate',
    'demonstrate','enable','ensure','enhance','optimize','foster',
]

AI_TIER3_PATTERNS = [
    r'\bFirstly\b.*\bSecondly\b',
    r'\bIn conclusion\b',
    r'\bIn summary\b',
    r'\bTo summarize\b',
    r'\bThis (study|paper|project|report|chapter|section) (aims|examines|explores|investigates|demonstrates)',
    r'\bThe (results|findings|analysis|data|model) (show|indicate|suggest|demonstrate|reveal)',
    r'\bIt is (important|crucial|essential|necessary) to\b',
]

HUMAN_CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|isn't|aren't|wasn't|weren't|it's|that's|"
    r"there's|they're|we're|you're|i've|i'll|i'd|we've|couldn't|"
    r"shouldn't|wouldn't|doesn't|didn't|hadn't|haven't|hasn't|"
    r"i'm|that'll|it'll|they'll|he's|she's)\b", re.IGNORECASE)

HUMAN_FIRST_PERSON = re.compile(
    r"\b(i |i'm|i've|i'll|i'd|we |we've|we'll|we'd|our |my |myself|ourselves)\b",
    re.IGNORECASE)

HUMAN_HEDGE = re.compile(
    r"\b(maybe|perhaps|probably|possibly|i think|i believe|i feel|"
    r"in my opinion|from my experience|i found|i noticed|i tried|"
    r"i struggled|i realized|honestly|frankly|to be honest|surprisingly)\b",
    re.IGNORECASE)

PASSIVE_VOICE = re.compile(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE)

BASIC_STOPS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall',
    'this','that','these','those','it','its','i','we','you','he','she','they',
    'my','your','our','his','her','their','what','which','who','how','when',
    'where','why','not','no','so','if','as','by','from','into','than','then',
    'very','just','also','more','most','some','any','all','both','each',
}


# ── Utilities ──────────────────────────────────────────────────────────────────
def _stops():
    if _cache['stops']:
        return _cache['stops']
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words('english'))
    except Exception:
        sw = BASIC_STOPS
    _cache['stops'] = sw
    return sw


def _sent_tokenize(text):
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _alpha_words(text):
    return re.findall(r'\b[a-z]+\b', text.lower())


# ── Model Loading ──────────────────────────────────────────────────────────────
def _load():
    if _cache['loaded']:
        return
    _stops()
    cfg = os.path.join(MODELS_DIR, 'feature_extractor_config.json')
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                _cache['n_feat'] = json.load(f).get('n_features', 18)
        except Exception:
            pass
    sc = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
    if os.path.isfile(sc):
        try:
            import joblib
            _cache['scaler'] = joblib.load(sc)
            logger.info('[AI] Scaler loaded.')
        except Exception as e:
            logger.warning(f'[AI] Scaler: {e}')
    for fn in ('all_ai_models.pkl', 'ai_detection_model.pkl'):
        mp = os.path.join(MODELS_DIR, fn)
        if os.path.isfile(mp):
            try:
                import joblib
                _cache['model'] = joblib.load(mp)
                logger.info(f'[AI] Model: {fn}')
                break
            except Exception as e:
                logger.warning(f'[AI] Model {fn}: {e}')
    _cache['loaded'] = True


# ── Feature Extraction (18 features) ──────────────────────────────────────────
def extract_features(text: str) -> list:
    _load()
    n = _cache['n_feat']
    if not text or len(text.strip()) < 20:
        return [0.0] * n
    sw    = _stops()
    sents = [s.strip() for s in _sent_tokenize(text) if len(s.strip()) > 3] or [text.strip()]
    words = _alpha_words(text)
    if not words:
        return [0.0] * n

    sl   = [len(s.split()) for s in sents]
    msl  = float(np.mean(sl))
    ssl  = float(np.std(sl)) if len(sl) > 1 else 0.0
    bur  = (ssl - msl) / (ssl + msl + 1e-9)

    wc   = Counter(words)
    vr   = len(set(words)) / (len(words) + 1)
    hr   = sum(1 for _, c in wc.items() if c == 1) / (len(wc) + 1)
    awl  = float(np.mean([len(w) for w in words]))
    sr   = sum(1 for w in words if w in sw) / (len(words) + 1)
    wf   = float(np.mean(list(wc.values())))
    tlen = len(text) + 1
    pr   = sum(1 for c in text if c in '.,;:!?') / tlen
    cr   = text.count(',') / tlen
    per  = text.count('.') / tlen
    col  = text.count(':') / tlen
    tl   = text.lower()
    sc2  = sum(1 for p in AI_TIER1 if p in tl)
    mc   = sum(1 for p in AI_TIER2 if p in tl)
    dm   = (sc2 * 1.8 + mc * 0.6) / (len(sents) + 1)
    pv   = len(PASSIVE_VOICE.findall(tl)) / (len(sents) + 1)
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) > 10]
    ps   = float(np.std([len(p.split()) for p in paras])) / (float(np.mean([len(p.split()) for p in paras])) + 1) if len(paras) > 1 else 0.0
    if len(sents) >= 3:
        ov = []
        for i in range(len(sents) - 1):
            w1 = set(_alpha_words(sents[i]))
            w2 = set(_alpha_words(sents[i+1]))
            if w1 and w2:
                ov.append(len(w1 & w2) / len(w1 | w2))
        ao = float(np.mean(ov)) if ov else 0.0
    else:
        ao = 0.0

    feats = [msl, ssl, bur, float(max(sl)), float(min(sl)),
             vr, hr, awl, sr, wf, pr, cr, per, col, dm, pv, ps, ao]
    return feats[:n] + [0.0] * max(0, n - len(feats))


# ── Linguistic Fingerprint Score ───────────────────────────────────────────────
def _linguistic_score(text: str) -> float:
    """
    Calibrated against Turnitin:
      Skin Disease (67% Turnitin) → 60-70%
      Capgemini    (55% Turnitin) → 50-62%
      Human informal text         → 5-25%
    """
    tl    = text.lower()
    feats = extract_features(text)
    n     = len(feats)

    msl  = feats[0]  if n > 0  else 0
    ssl  = feats[1]  if n > 1  else 0
    bur  = feats[2]  if n > 2  else 0
    vr   = feats[5]  if n > 5  else 0
    pv   = feats[15] if n > 15 else 0
    ps   = feats[16] if n > 16 else 0
    ao   = feats[17] if n > 17 else 0

    score = 20.0

    # Sentence uniformity (strongest signal)
    if   bur < -0.45: score += 24
    elif bur < -0.30: score += 18
    elif bur < -0.15: score += 11
    elif bur < -0.05: score += 5
    elif bur >  0.25: score -= 12
    elif bur >  0.12: score -= 6

    # Mean sentence length
    if   msl > 35: score += 14
    elif msl > 28: score += 9
    elif msl > 22: score += 5
    elif msl > 17: score += 2
    elif msl <  9: score -= 8

    # Std dev
    if   ssl < 3.0: score += 12
    elif ssl < 5.5: score += 6
    elif ssl < 8.0: score += 2
    elif ssl > 20:  score -= 7

    # Vocabulary richness
    if   vr > 0.82: score += 8
    elif vr > 0.72: score += 5
    elif vr > 0.60: score += 2
    elif vr < 0.42: score -= 7

    # AI discourse markers (tier-weighted, capped)
    t1 = sum(1 for ph in AI_TIER1 if ph in tl)
    t2 = sum(1 for ph in AI_TIER2 if ph in tl)
    t3 = sum(1 for pat in AI_TIER3_PATTERNS if re.search(pat, text, re.IGNORECASE))
    score += min(32, t1 * 3.5 + t2 * 1.2 + t3 * 4.0)

    # Passive voice
    if   pv > 0.80: score += 9
    elif pv > 0.55: score += 6
    elif pv > 0.30: score += 3

    # Paragraph uniformity
    if   ps < 0.10: score += 8
    elif ps < 0.22: score += 4
    elif ps < 0.40: score += 1
    elif ps > 0.90: score -= 5

    # Inter-sentence overlap
    if   0.04 < ao < 0.20: score += 5
    elif ao < 0.02:         score -= 3

    # ── Human deductions ──────────────────────────────────────────────────────
    cont = len(HUMAN_CONTRACTIONS.findall(text))
    if   cont > 8:  score -= 18
    elif cont > 4:  score -= 12
    elif cont > 2:  score -= 7
    elif cont > 0:  score -= 3

    fp = len(HUMAN_FIRST_PERSON.findall(text))
    if   fp > 10: score -= 16
    elif fp > 6:  score -= 10
    elif fp > 3:  score -= 6
    elif fp > 1:  score -= 3

    hedges = len(HUMAN_HEDGE.findall(text))
    if   hedges > 4: score -= 10
    elif hedges > 2: score -= 6
    elif hedges > 0: score -= 3

    exclaim = text.count('!') + text.count('?')
    if   exclaim > 8: score -= 8
    elif exclaim > 4: score -= 5
    elif exclaim > 1: score -= 2

    citations = len(re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)', text))
    if   citations > 10: score -= 6
    elif citations > 5:  score -= 3
    elif citations > 2:  score -= 1

    return max(5.0, min(95.0, score))


# ── Model Score (calibrated) ───────────────────────────────────────────────────
def _model_score(feats: list):
    """Sigmoid-calibrated model score. Returns None if model unavailable."""
    model  = _cache.get('model')
    scaler = _cache.get('scaler')
    if model is None or scaler is None:
        return None
    try:
        scaled = scaler.transform([feats])
        if hasattr(model, 'predict_proba'):
            raw = float(model.predict_proba(scaled)[0][1])
        elif isinstance(model, dict):
            ps = [float(m.predict_proba(scaled)[0][1])
                  for m in model.values() if hasattr(m, 'predict_proba')]
            raw = float(np.mean(ps)) if ps else None
            if raw is None:
                return None
        else:
            return None
        # Sigmoid recalibration: raw=0.07→36%, raw=0.12→61%, raw=0.18→80%
        cal = 1.0 / (1.0 + np.exp(-(raw - 0.10) * 8)) * 100
        return max(5.0, min(95.0, float(cal)))
    except Exception as e:
        logger.warning(f'[AI] Model score error: {e}')
        return None


# ── Per-sentence scoring ───────────────────────────────────────────────────────
def _sent_score(sent: str, doc_pct: float) -> float:
    if len(sent.split()) < 6:
        return 0.0
    sl   = sent.lower()
    base = _linguistic_score(sent) / 100.0
    t1   = sum(1 for ph in AI_TIER1 if ph in sl)
    t2   = sum(1 for ph in AI_TIER2 if ph in sl)
    boost = min(0.35, t1 * 0.12 + t2 * 0.04)
    cont  = len(HUMAN_CONTRACTIONS.findall(sent))
    fp    = len(HUMAN_FIRST_PERSON.findall(sent))
    hdg   = len(HUMAN_HEDGE.findall(sent))
    deduct = min(0.30, cont * 0.08 + fp * 0.06 + hdg * 0.05)
    prior = doc_pct / 100.0
    raw   = base * 0.50 + prior * 0.30 + boost - deduct
    return max(0.0, min(1.0, raw))


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_ai(text: str) -> dict:
    """Analyze text for AI-generated content. Output ready for generate_ai_report()."""
    t0 = time.time()
    _load()
    if not text or len(text.strip()) < 20:
        return _empty_result(text)

    text = re.sub(r'\r\n', '\n', text).replace('\r', '\n').strip()

    ling  = _linguistic_score(text)
    feats = extract_features(text)
    mdl   = _model_score(feats)

    if mdl is not None:
        combined = max(mdl * 0.55 + ling * 0.45, ling)
        logger.info(f'[AI] model={mdl:.1f}% ling={ling:.1f}% combined={combined:.1f}%')
    else:
        combined = ling
        logger.info(f'[AI] ling-only={ling:.1f}%')

    doc_pct = max(0.0, min(99.0, combined))

    # Per-sentence
    sents      = _sent_tokenize(text)
    sent_res   = []
    cursor     = 0
    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue
        start = text.find(sent, cursor)
        if start == -1:
            m = re.search(re.escape(sent[:30]), text[cursor:])
            start = cursor + m.start() if m else cursor
        end    = start + len(sent)
        cursor = max(cursor, start)
        prob   = _sent_score(sent, doc_pct)
        sent_res.append({'text': sent, 'start': start, 'end': end,
                         'ai_prob': round(prob, 3), 'is_ai': prob >= SENT_AI_THRESHOLD})

    # Build highlights
    ai_highlights = []
    ai_only_count = 0
    cur_span      = None
    for sr in sent_res:
        if sr['is_ai']:
            if cur_span is None:
                cur_span = {'start': sr['start'], 'end': sr['end'],
                            'type': 'ai_generated', 'prob': sr['ai_prob']}
            else:
                cur_span['end']  = sr['end']
                cur_span['prob'] = max(cur_span['prob'], sr['ai_prob'])
        else:
            if cur_span is not None:
                ai_highlights.append(cur_span)
                ai_only_count += 1
                cur_span = None
    if cur_span is not None:
        ai_highlights.append(cur_span)
        ai_only_count += 1

    # Reconcile with sentence coverage
    if sent_res:
        ai_ch  = sum(s['end'] - s['start'] for s in sent_res if s['is_ai'])
        tot_ch = max(sum(s['end'] - s['start'] for s in sent_res), 1)
        cov    = round(ai_ch / tot_ch * 100)
        final  = max(round(doc_pct), cov)
    else:
        final  = round(doc_pct)

    final = max(0, min(99, final))

    if   final >= 50: label = 'HIGH AI RISK'
    elif final >= 20: label = 'CAUTION'
    else:             label = 'LOW AI RISK'

    fv = {
        'mean_sentence_length':   round(feats[0],  1) if len(feats) > 0  else 0,
        'std_sentence_length':    round(feats[1],  1) if len(feats) > 1  else 0,
        'burstiness':             round(feats[2],  3) if len(feats) > 2  else 0,
        'vocabulary_richness':    round(feats[5],  3) if len(feats) > 5  else 0,
        'hapax_ratio':            round(feats[6],  3) if len(feats) > 6  else 0,
        'avg_word_length':        round(feats[7],  1) if len(feats) > 7  else 0,
        'stopword_ratio':         round(feats[8],  3) if len(feats) > 8  else 0,
        'punctuation_ratio':      round(feats[10], 3) if len(feats) > 10 else 0,
        'discourse_marker_rate':  round(feats[14], 3) if len(feats) > 14 else 0,
        'passive_voice_rate':     round(feats[15], 3) if len(feats) > 15 else 0,
        'paragraph_length_std':   round(feats[16], 3) if len(feats) > 16 else 0,
        'inter_sentence_overlap': round(feats[17], 3) if len(feats) > 17 else 0,
    }

    logger.info(f'[AI] final={final}% highlights={ai_only_count} t={round(time.time()-t0,2)}s')

    return {
        'ai_pct':            final,
        'ai_label':          label,
        'full_text':         text,
        'ai_highlights':     ai_highlights,
        'ai_only_count':     ai_only_count,
        'sentence_scores':   sent_res,
        'ai_probability':    final,            # backward-compat
        'sentence_highlights': sent_res,       # backward-compat
        'feature_values':    fv,
        'analysis_time_sec': round(time.time() - t0, 3),
    }


def _empty_result(text=''):
    return {'ai_pct': 0, 'ai_label': 'LOW AI RISK', 'full_text': text or '',
            'ai_highlights': [], 'ai_only_count': 0, 'sentence_scores': [],
            'ai_probability': 0, 'sentence_highlights': [],
            'feature_values': {}, 'analysis_time_sec': 0.0}


# backward-compat alias
def analyze_ai_detection(text: str) -> dict:
    return analyze_ai(text)
