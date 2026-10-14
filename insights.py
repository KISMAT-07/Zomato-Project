"""Prediction explanations, phrase summaries and validated batch inference."""
import html
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentiment import normalize_review

MAX_ROWS = 2000
MAX_BYTES = 5 * 1024 * 1024


def explain_review(model, review, limit=10):
    vector = model.named_steps['tfidf'].transform([review])
    classifier = model.named_steps['classifier']
    label = model.predict([review])[0]
    index = list(classifier.classes_).index(label)
    names = model.named_steps['tfidf'].get_feature_names_out()
    contributions = vector.data * classifier.coef_[index, vector.indices]
    order = np.argsort(np.abs(contributions))[::-1][:limit]
    return pd.DataFrame([{'Phrase': names[vector.indices[i]], 'Contribution': float(contributions[i])} for i in order])


def highlight_review(review, explanation):
    # Mark the normalized representation: these are the actual model features.
    text = normalize_review(review)
    supporters = {row.Phrase for row in explanation.itertuples() if row.Contribution > 0}
    pattern = '|'.join(re.escape(p) for p in sorted(supporters, key=len, reverse=True))
    if not pattern:
        return html.escape(text)
    parts, start = [], 0
    for match in re.finditer(r'(?<!\w)(?:' + pattern + r')(?!\w)', text):
        parts.extend([html.escape(text[start:match.start()]), '<mark>' + html.escape(match.group()) + '</mark>'])
        start = match.end()
    parts.append(html.escape(text[start:]))
    return ''.join(parts)


def common_phrases(reviews, limit=8):
    texts = [normalize_review(x) for x in reviews if isinstance(x, str)]
    if not any(texts):
        return pd.DataFrame(columns=['Phrase', 'Reviews mentioning phrase'])
    try:
        vectorizer = CountVectorizer(ngram_range=(2, 2), binary=True, min_df=2)
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame(columns=['Phrase', 'Reviews mentioning phrase'])
    return pd.DataFrame({'Phrase': vectorizer.get_feature_names_out(),
        'Reviews mentioning phrase': np.asarray(matrix.sum(axis=0)).ravel()}).sort_values(
        ['Reviews mentioning phrase', 'Phrase'], ascending=[False, True]).head(limit)


def analyze_batch(model, frame, column):
    if frame.empty:
        raise ValueError('The CSV contains no data rows.')
    if len(frame) > MAX_ROWS:
        raise ValueError(f'Please upload at most {MAX_ROWS:,} rows.')
    if column not in frame:
        raise ValueError('Choose a review column from your CSV.')
    reserved = {'Predicted sentiment', 'Top model probability', 'Analysis status'}
    if reserved.intersection(frame.columns):
        raise ValueError('Remove or rename existing result columns before analyzing this CSV.')
    result = frame.copy()
    texts = result[column].fillna('').astype(str)
    if texts.str.len().gt(5000).any():
        raise ValueError('Each review must be 5,000 characters or fewer.')
    meaningful = texts.map(normalize_review).str.len().gt(0).to_numpy()
    known = np.asarray(model.named_steps['tfidf'].transform(texts).getnnz(axis=1)).ravel() > 0
    valid = meaningful & known
    result['Predicted sentiment'] = ''
    result['Top model probability'] = np.nan
    result['Analysis status'] = np.where(~meaningful, 'Empty or no meaningful words', 'Outside model vocabulary')
    if valid.any():
        result.loc[valid, 'Predicted sentiment'] = model.predict(texts[valid])
        result.loc[valid, 'Top model probability'] = model.predict_proba(texts[valid]).max(axis=1)
        result.loc[valid, 'Analysis status'] = 'Analyzed'
    return result


def safe_csv(frame):
    # Prevent review text from being interpreted as spreadsheet formulas.
    result = frame.copy()
    for column in result.select_dtypes(include=['object', 'string']):
        result[column] = result[column].map(lambda value: "'" + value if isinstance(value, str)
            and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value)
    return result.to_csv(index=False).encode('utf-8-sig')
