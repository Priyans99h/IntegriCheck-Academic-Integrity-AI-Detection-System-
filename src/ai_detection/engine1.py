"""
IntegriCheck — AI Detection Engine v5
======================================
DROP-IN REPLACEMENT for: src/ai_detection/engine.py

WHAT THIS ENGINE DOES:
  1. Accepts submitted text
  2. Extracts 18 statistical + linguistic features
  3. Runs trained ensemble model (or calibrated heuristic fallback)
  4. Returns per-sentence AI probability + character highlights
  5. Output dict is PERFECTLY shaped for report_generator.py

OUTPUT DICT SCHEMA (what analyze_ai returns):
  {
    'ai_pct':          int,         # 0–100 overall AI probability
    'ai_label':        str,         # 'HIGH AI RISK' | 'CAUTION' | 'LOW AI RISK'
    'full_text':       str,         # original submitted text (pass-through)
    'ai_highlights': [              # per-sentence char spans for AI text
        {
            'start': int,           # char offset in full_text
            'end':   int,           # char offset in full_text
            'type':  str,           # 'ai_generated' | 'ai_paraphrased'
            'prob':  float,         # 0.0–1.0 sentence-level AI probability
        },
        ...
    ],
    'ai_only_count':   int,         # count of 'ai_generated' spans (for Detection Groups badge)
    'sentence_scores': [            # per-sentence breakdown
        {
            'text':    str,
            'ai_prob': float,
            'is_ai':   bool,
            'start':   int,
            'end':     int,
        },
        ...
    ],
    'feature_values': {             # extracted features (for analysis page)
        'mean_sentence_length':  float,
        'vocabulary_richness':   float,
        'discourse_marker_rate': float,
        'burstiness':            float,
        'passive_voice_rate':    float,
        ...
    },
    'analysis_time_sec': float,
  }

HOW TO CALL (from flask_app/app.py):
    from src.ai_detection.engine import analyze_ai

    result = analyze_ai(text)
    # result is ready to pass directly to generate_ai_report(result, path)
"""

import os
import re
import json
import time
import logging
from collections import Counter

import numpy as np

logger = logging.getLogger('integricheck.ai_detection')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')

# ── Model cache ────────────────────────────────────────────────────────────────
_cache = {
    'model':   None,
    'scaler':  None,
    'stops':   None,
    'n_feat':  18,
    'loaded':  False,
}

# ── Thresholds ────────────────────────────────────────────────────────────────
SENTENCE_AI_THRESHOLD  = 0.60    # per-sentence probability → mark as AI (raised: only flag clearly AI)
DOCUMENT_AI_THRESHOLD  = 0.55    # overall document threshold
MIN_SENTENCE_WORDS     = 8       # skip very short sentences in per-sentence analysis

# ══════════════════════════════════════════════════════════════════════════════
# LINGUISTIC SIGNAL LISTS
# ══════════════════════════════════════════════════════════════════════════════

# Strong AI signals — phrases almost exclusively used by LLMs in formal text
AI_PHRASES_STRONG = [
    'delve into', 'delves into', 'delving into',
    'embark on', 'embarks on', 'embarking on',
    'harness the power', 'harnessing the power',
    'seamlessly integrates', 'seamlessly integrated',
    'in the realm of', 'in the landscape of',
    'it is worth noting that', 'it is important to note that',
    'needless to say', 'it goes without saying',
    'this underscores the', 'this highlights the',
    'multifaceted nature', 'multifaceted approach',
    'pivotal role', 'plays a pivotal',
    'a testament to', 'stands as a testament',
    'foster a culture', 'fostering innovation',
    'at its core', 'as we navigate',
    'by leveraging', 'holistic approach',
    'nuanced understanding', 'comprehensive overview',
    'in today\'s rapidly evolving', 'in today\'s fast-paced',
    'when it comes to', 'in order to ensure',
    'it\'s important to note', 'it is crucial to',
]

# Moderate AI signals — elevated connectives typical in LLM output
AI_PHRASES_MODERATE = [
    'furthermore', 'moreover', 'additionally', 'consequently',
    'in conclusion', 'to summarize', 'in summary', 'to conclude',
    'on the other hand', 'as mentioned', 'as noted above',
    'in other words', 'that being said', 'having said that',
    'in this context', 'with that in mind', 'to this end',
    'robust', 'crucial', 'intricate', 'synergy', 'paradigm',
    'leverage', 'innovative', 'transformative', 'groundbreaking',
    'state-of-the-art', 'cutting-edge',
]

# Strong HUMAN signals — markers of genuine human authorship
HUMAN_SIGNALS_STRONG = [
    r"\bi\s+(?:think|believe|feel|found|noticed|tried|struggled|realized|was|have)\b",
    r"\bwe\s+(?:tried|found|noticed|faced|realized)\b",
    r"\bto be honest\b",
    r"\bhonestly\b",
    r"\bin my opinion\b",
    r"\bfrom my experience\b",
    r"\bactually\b.*\bbut\b",
]

HUMAN_CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|isn't|aren't|wasn't|weren't|it's|that's|"
    r"there's|they're|we're|you're|i've|i'll|i'd|we've|couldn't|"
    r"shouldn't|wouldn't)\b",
    re.IGNORECASE
)

FIRST_PERSON = re.compile(
    r"\b(i |i'm|i've|i'll|i'd|we |we've|we'll|our |my |myself|ourselves)\b",
    re.IGNORECASE
)

PASSIVE_VOICE = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
    re.IGNORECASE
)

BASIC_STOPS = {
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','could','should','may','might','shall',
    'this','that','these','those','it','its','i','we','you','he','she','they',
    'my','your','our','his','her','their','what','which','who','how','when',
    'where','why','not','no','so','if','as','by','from','into','than','then',
    'very','just','also','more','most','some','any','all','both','each',
    'few','many','much','other','such','only','own','same','too','well',
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _get_stopwords():
    if _cache['stops']:
        return _cache['stops']
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words('english'))
    except Exception:
        sw = BASIC_STOPS
    _cache['stops'] = sw
    return sw


def _tokenize_sentences(text):
    try:
        import nltk
        return nltk.sent_tokenize(text)
    except Exception:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]


def _tokenize_words(text):
    try:
        import nltk
        return nltk.word_tokenize(text.lower())
    except Exception:
        return re.findall(r"\b[a-z']+\b", text.lower())


def _alpha_words(text):
    """Alphabetic tokens only."""
    return re.findall(r'\b[a-z]+\b', text.lower())


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_model():
    if _cache['loaded']:
        return

    # Stopwords
    _get_stopwords()

    # Feature config
    cfg_path = os.path.join(MODELS_DIR, 'feature_extractor_config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            _cache['n_feat'] = cfg.get('n_features', 18)
        except Exception:
            pass

    # Scaler
    scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.pkl')
    if os.path.isfile(scaler_path):
        try:
            import joblib
            _cache['scaler'] = joblib.load(scaler_path)
            logger.info('[AI] Feature scaler loaded.')
        except Exception as e:
            logger.warning(f'[AI] Scaler load failed: {e}')

    # Model (ensemble or single)
    for model_file in ('all_ai_models.pkl', 'ai_detection_model.pkl'):
        model_path = os.path.join(MODELS_DIR, model_file)
        if os.path.isfile(model_path):
            try:
                import joblib
                _cache['model'] = joblib.load(model_path)
                logger.info(f'[AI] Model loaded: {model_file}')
                break
            except Exception as e:
                logger.warning(f'[AI] Model load failed ({model_file}): {e}')

    _cache['loaded'] = True


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(text: str) -> list:
    """
    Extract 18 statistical and linguistic features from text.

    Feature index map:
      0  mean_sentence_length
      1  std_sentence_length
      2  burstiness          (sentence length variability index)
      3  max_sentence_length
      4  min_sentence_length
      5  vocabulary_richness  (type-token ratio)
      6  hapax_ratio          (words appearing exactly once / vocab)
      7  avg_word_length
      8  stopword_ratio
      9  mean_word_frequency
      10 punctuation_ratio
      11 comma_ratio
      12 period_ratio
      13 colon_ratio
      14 discourse_marker_rate
      15 passive_voice_rate
      16 paragraph_length_std
      17 inter_sentence_overlap

    Returns list of 18 floats (padded with 0.0 if text too short).
    """
    _load_model()
    n = _cache.get('n_feat', 18)

    if not text or len(text.strip()) < 20:
        return [0.0] * n

    sw       = _get_stopwords()
    sents    = [s.strip() for s in _tokenize_sentences(text) if len(s.strip()) > 3]
    sents    = sents if sents else [text.strip()]
    words_a  = _alpha_words(text)
    words_t  = _tokenize_words(text)
    words_t  = [w for w in words_t if w.isalpha()]

    if not words_a:
        return [0.0] * n

    # ── Sentence stats ─────────────────────────────────────────────────────────
    sent_lens  = [len(s.split()) for s in sents]
    mean_sl    = float(np.mean(sent_lens))
    std_sl     = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
    max_sl     = float(max(sent_lens))
    min_sl     = float(min(sent_lens))
    burstiness = (std_sl - mean_sl) / (std_sl + mean_sl + 1e-9)

    # ── Vocabulary stats ───────────────────────────────────────────────────────
    wc         = Counter(words_a)
    vocab_r    = len(set(words_a)) / (len(words_a) + 1)
    hapax_r    = sum(1 for w, c in wc.items() if c == 1) / (len(wc) + 1)
    avg_wl     = float(np.mean([len(w) for w in words_a]))
    stop_r     = sum(1 for w in words_a if w in sw) / (len(words_a) + 1)
    wfreq      = float(np.mean(list(wc.values())))

    # ── Punctuation stats ──────────────────────────────────────────────────────
    tlen       = len(text) + 1
    punct_r    = sum(1 for c in text if c in '.,;:!?') / tlen
    comma_r    = text.count(',') / tlen
    period_r   = text.count('.') / tlen
    colon_r    = text.count(':') / tlen

    # ── Discourse markers ──────────────────────────────────────────────────────
    text_l = text.lower()
    strong_count   = sum(1 for ph in AI_PHRASES_STRONG   if ph in text_l)
    moderate_count = sum(1 for ph in AI_PHRASES_MODERATE if ph in text_l)
    dm_rate        = (strong_count * 1.8 + moderate_count * 0.6) / (len(sents) + 1)

    # ── Passive voice ──────────────────────────────────────────────────────────
    passive_count = len(PASSIVE_VOICE.findall(text_l))
    passive_r     = passive_count / (len(sents) + 1)

    # ── Paragraph uniformity ───────────────────────────────────────────────────
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip().split()) > 10]
    if len(paras) > 1:
        para_lens = [len(p.split()) for p in paras]
        para_std  = float(np.std(para_lens)) / (float(np.mean(para_lens)) + 1)
    else:
        para_std = 0.0

    # ── Inter-sentence word overlap ────────────────────────────────────────────
    if len(sents) >= 3:
        overlaps = []
        for i in range(len(sents) - 1):
            w1 = set(_alpha_words(sents[i]))
            w2 = set(_alpha_words(sents[i+1]))
            if w1 and w2:
                overlaps.append(len(w1 & w2) / len(w1 | w2))
        avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    else:
        avg_overlap = 0.0

    # Trim/pad to exactly n features
    feats = [
        mean_sl, std_sl, burstiness, max_sl, min_sl,
        vocab_r, hapax_r, avg_wl, stop_r, wfreq,
        punct_r, comma_r, period_r, colon_r,
        dm_rate, passive_r,
        para_std, avg_overlap,
    ]
    return feats[:n] + [0.0] * max(0, n - len(feats))


# ══════════════════════════════════════════════════════════════════════════════
# AI PROBABILITY — HEURISTIC FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_probability(text: str, feats: list) -> float:
    """
    Calibrated heuristic AI probability when trained model is unavailable.

    Design principles:
    - Starts at 28 (baseline — most text has some AI-like properties)
    - Adds evidence-based points for AI signals
    - SUBTRACTS points for strong human signals
    - Clamps 5–95 to never be absolute
    """
    n      = len(feats)
    mean_sl    = feats[0]  if n > 0  else 0
    std_sl     = feats[1]  if n > 1  else 0
    burstiness = feats[2]  if n > 2  else 0
    vocab_r    = feats[5]  if n > 5  else 0
    dm_rate    = feats[14] if n > 14 else 0
    passive_r  = feats[15] if n > 15 else 0
    para_std   = feats[16] if n > 16 else 0

    score = 28.0

    # ── Sentence length uniformity ─────────────────────────────────────────────
    # AI: very consistent sentence lengths → low burstiness
    if burstiness < -0.40:   score += 20
    elif burstiness < -0.25: score += 12
    elif burstiness < -0.08: score += 5
    elif burstiness > 0.20:  score -= 10   # human: varied lengths
    elif burstiness > 0.08:  score -= 4

    # ── Mean sentence length ───────────────────────────────────────────────────
    if mean_sl > 34:    score += 12
    elif mean_sl > 26:  score += 6
    elif mean_sl > 18:  score += 2
    elif mean_sl < 8:   score -= 7

    # ── Vocabulary richness ────────────────────────────────────────────────────
    if vocab_r > 0.85:   score += 7
    elif vocab_r > 0.75: score += 4
    elif vocab_r < 0.45: score -= 6

    # ── Sentence length consistency ────────────────────────────────────────────
    if std_sl < 3:    score += 10
    elif std_sl < 5:  score += 5
    elif std_sl > 18: score -= 6

    # ── Discourse markers density ──────────────────────────────────────────────
    if dm_rate > 1.0:    score += 18
    elif dm_rate > 0.7:  score += 12
    elif dm_rate > 0.4:  score += 7
    elif dm_rate > 0.2:  score += 3

    # ── Passive voice ──────────────────────────────────────────────────────────
    if passive_r > 0.7:    score += 7
    elif passive_r > 0.45: score += 4

    # ── Paragraph uniformity ───────────────────────────────────────────────────
    if para_std < 0.12:   score += 7
    elif para_std < 0.25: score += 3
    elif para_std > 0.80: score -= 5

    # ── Human signal deductions ────────────────────────────────────────────────
    text_l = text.lower()

    # Contractions
    contractions = len(HUMAN_CONTRACTIONS.findall(text))
    if contractions > 5:  score -= 10
    elif contractions > 2: score -= 5

    # First-person
    fp_count = len(FIRST_PERSON.findall(text))
    if fp_count > 6:   score -= 10
    elif fp_count > 3: score -= 5
    elif fp_count > 1: score -= 2

    # Informal punctuation
    informal = text.count('!') + text.count('?')
    if informal > 6:  score -= 6
    elif informal > 3: score -= 3

    # Human-specific phrases
    human_hits = sum(1 for pat in HUMAN_SIGNALS_STRONG
                     if re.search(pat, text_l))
    score -= human_hits * 4

    # Typos / spelling errors (proxy: words not in a common word set)
    # If document has a few apparent typos, it's probably human
    common_words = {'teh', 'recieve', 'occured', 'seperate', 'definately',
                    'goverment', 'sucessful', 'untill', 'accomodate', 'acheive'}
    typo_count = sum(1 for w in _alpha_words(text) if w in common_words)
    if typo_count >= 2:
        score -= 8

    return max(5.0, min(95.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# PER-SENTENCE AI SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _per_sentence_probability(text: str) -> list:
    """
    Assign AI probability to each sentence.

    Returns list of:
        {text, start, end, ai_prob, is_ai}
    """
    sents   = _tokenize_sentences(text)
    results = []
    cursor  = 0
    model   = _cache.get('model')
    scaler  = _cache.get('scaler')

    for sent in sents:
        sent = sent.strip()
        if not sent:
            continue

        start = text.find(sent, cursor)
        if start == -1:
            m = re.search(re.escape(sent[:30]), text[cursor:])
            start = cursor + m.start() if m else cursor
        end = start + len(sent)
        cursor = max(cursor, start)

        word_count = len(sent.split())
        if word_count < MIN_SENTENCE_WORDS:
            results.append({
                'text':    sent, 'start': start, 'end': end,
                'ai_prob': 0.0,  'is_ai': False,
            })
            continue

        # Use model if available
        if model is not None and scaler is not None:
            try:
                feats  = extract_features(sent)
                scaled = scaler.transform([feats])
                # Handle single or ensemble model
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(scaled)[0][1])
                elif isinstance(model, dict):
                    # Our 'all_ai_models.pkl' may be a dict of models
                    probs = []
                    for m_name, m_obj in model.items():
                        if hasattr(m_obj, 'predict_proba'):
                            probs.append(float(m_obj.predict_proba(scaled)[0][1]))
                    prob = float(np.mean(probs)) if probs else 0.5
                else:
                    prob = 0.5
            except Exception:
                prob = _heuristic_probability(sent, extract_features(sent)) / 100.0
        else:
            # Heuristic fallback — sentence-level needs a lighter version
            feats = extract_features(sent)
            raw   = _heuristic_probability(sent, feats)

            # Adjust sentence-level: boost if strong AI phrases detected
            sent_l = sent.lower()
            strong_hits   = sum(1 for ph in AI_PHRASES_STRONG   if ph in sent_l)
            moderate_hits = sum(1 for ph in AI_PHRASES_MODERATE if ph in sent_l)
            boost = min(25, strong_hits * 10 + moderate_hits * 3)

            # Reduce if human signals
            fp    = len(FIRST_PERSON.findall(sent))
            cont  = len(HUMAN_CONTRACTIONS.findall(sent))
            deduct = min(20, fp * 5 + cont * 4)

            raw = raw + boost - deduct
            prob = max(5.0, min(95.0, raw)) / 100.0

        is_ai = prob >= SENTENCE_AI_THRESHOLD
        results.append({
            'text':    sent,
            'start':   start,
            'end':     end,
            'ai_prob': round(prob, 3),
            'is_ai':   is_ai,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ai(text: str) -> dict:
    """
    Main entry point. Analyzes text for AI-generated content.

    Parameters
    ----------
    text : str
        The submitted document text.

    Returns
    -------
    dict — ready to pass directly to generate_ai_report()
    """
    t0 = time.time()
    _load_model()

    if not text or len(text.strip()) < 20:
        return _empty_result(text)

    text = re.sub(r'\r\n', '\n', text).strip()

    # ── Document-level features + probability ─────────────────────────────────
    feats = extract_features(text)
    model  = _cache.get('model')
    scaler = _cache.get('scaler')

    if model is not None and scaler is not None:
        try:
            scaled = scaler.transform([feats])
            if hasattr(model, 'predict_proba'):
                doc_prob = float(model.predict_proba(scaled)[0][1]) * 100
            elif isinstance(model, dict):
                probs = []
                for m_name, m_obj in model.items():
                    if hasattr(m_obj, 'predict_proba'):
                        probs.append(float(m_obj.predict_proba(scaled)[0][1]))
                doc_prob = (float(np.mean(probs)) * 100) if probs else 50.0
            else:
                doc_prob = 50.0
        except Exception:
            doc_prob = _heuristic_probability(text, feats)
    else:
        doc_prob = _heuristic_probability(text, feats)

    ai_pct = max(0, min(99, round(doc_prob)))

    # ── Per-sentence analysis ─────────────────────────────────────────────────
    sentence_scores = _per_sentence_probability(text)

    # ── Build AI highlights ───────────────────────────────────────────────────
    ai_highlights = []
    ai_only_count = 0

    # Group consecutive AI sentences into spans
    current_span = None
    for ss in sentence_scores:
        if ss['is_ai']:
            if current_span is None:
                current_span = {'start': ss['start'], 'end': ss['end'],
                                'type': 'ai_generated', 'prob': ss['ai_prob']}
            else:
                current_span['end']  = ss['end']
                current_span['prob'] = max(current_span['prob'], ss['ai_prob'])
        else:
            if current_span is not None:
                ai_highlights.append(current_span)
                ai_only_count += 1
                current_span = None
    if current_span is not None:
        ai_highlights.append(current_span)
        ai_only_count += 1

    # ── AI label ──────────────────────────────────────────────────────────────
    if ai_pct >= 50:
        ai_label = 'HIGH AI RISK'
    elif ai_pct >= 20:
        ai_label = 'CAUTION'
    else:
        ai_label = 'LOW AI RISK'

    # ── Feature summary (human-readable) ─────────────────────────────────────
    feature_values = {
        'mean_sentence_length':   round(feats[0], 1) if len(feats) > 0  else 0,
        'std_sentence_length':    round(feats[1], 1) if len(feats) > 1  else 0,
        'burstiness':             round(feats[2], 3) if len(feats) > 2  else 0,
        'vocabulary_richness':    round(feats[5], 3) if len(feats) > 5  else 0,
        'hapax_ratio':            round(feats[6], 3) if len(feats) > 6  else 0,
        'avg_word_length':        round(feats[7], 1) if len(feats) > 7  else 0,
        'stopword_ratio':         round(feats[8], 3) if len(feats) > 8  else 0,
        'punctuation_ratio':      round(feats[10],3) if len(feats) > 10 else 0,
        'discourse_marker_rate':  round(feats[14],3) if len(feats) > 14 else 0,
        'passive_voice_rate':     round(feats[15],3) if len(feats) > 15 else 0,
        'paragraph_length_std':   round(feats[16],3) if len(feats) > 16 else 0,
        'inter_sentence_overlap': round(feats[17],3) if len(feats) > 17 else 0,
    }

    return {
        'ai_pct':            ai_pct,
        'ai_label':          ai_label,
        'full_text':         text,
        'ai_highlights':     ai_highlights,
        'ai_only_count':     ai_only_count,
        'sentence_scores':   sentence_scores,
        'feature_values':    feature_values,
        'analysis_time_sec': round(time.time() - t0, 3),
    }


def _empty_result(text=''):
    return {
        'ai_pct':            0,
        'ai_label':          'LOW AI RISK',
        'full_text':         text or '',
        'ai_highlights':     [],
        'ai_only_count':     0,
        'sentence_scores':   [],
        'feature_values':    {},
        'analysis_time_sec': 0.0,
    }
