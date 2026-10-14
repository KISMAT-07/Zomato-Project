"""Zomato portfolio dashboard. Run with streamlit run app.py."""
import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sentiment import normalize_review

ROOT = Path(__file__).resolve().parent
COLORS = {'Positive': '#238B68', 'Neutral': '#E6A23C', 'Negative': '#E23744'}
st.set_page_config(page_title='Zomato | Review Explorer', page_icon='🍽️', layout='wide', initial_sidebar_state='collapsed')

@st.cache_data
def load_data():
    frame = pd.read_csv(ROOT / 'data/zomato_cleaned_step1.csv')
    frame['Time'] = pd.to_datetime(frame['Time'], errors='coerce')
    return frame

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'models/portfolio_sentiment_model.joblib')

@st.cache_data
def load_results():
    return json.loads((ROOT / 'data/portfolio_model_results.json').read_text())

st.title('Zomato Review Explorer')
st.markdown('Restaurant insights, customer sentiment, and a review classifier built with machine learning.')
st.caption('Portfolio project by Puran • Historical dataset • Independent project; not affiliated with Zomato')
try:
    data = load_data()
except (OSError, ValueError) as exc:
    st.error(f'The restaurant dataset could not be loaded: {exc}')
    st.stop()

with st.sidebar:
    st.header('Explore the dataset')
    cuisines = st.multiselect('Primary cuisine', sorted(data['Primary_Cuisine'].dropna().unique()))
    choices = data if not cuisines else data[data['Primary_Cuisine'].isin(cuisines)]
    restaurants = st.multiselect('Restaurant', sorted(choices['Restaurant'].dropna().unique()))
    min_rating, max_rating = st.slider('Review rating', 1.0, 5.0, (1.0, 5.0), 0.5)
    st.caption('Filters apply to the overview and restaurant explorer.')
    st.link_button('View source on GitHub', 'https://github.com/purans07/Zomato-Project')
filtered = choices[choices['Rating_num'].between(min_rating, max_rating)]
if restaurants:
    filtered = filtered[filtered['Restaurant'].isin(restaurants)]

overview, explorer, predict, evaluation = st.tabs(['Overview', 'Restaurant explorer', 'Try a review', 'Model & methodology'])
with overview:
    st.subheader('What do customers say?')
    if filtered.empty:
        st.info('No reviews match these filters. Try a wider selection.')
    else:
        a, b, c, d = st.columns(4)
        a.metric('Reviews', f'{len(filtered):,}')
        b.metric('Restaurants', filtered['Restaurant'].nunique())
        c.metric('Average rating', f"{filtered['Rating_num'].mean():.2f} / 5")
        d.metric('Positive reviews', f"{filtered['Sentiment'].eq('Positive').mean():.0%}")
        left, right = st.columns(2)
        counts = filtered['Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
        left.plotly_chart(px.pie(counts, names='Sentiment', values='Reviews', hole=.55,
            color='Sentiment', color_discrete_map=COLORS, title='Sentiment from review ratings'), use_container_width=True)
        right.plotly_chart(px.histogram(filtered, x='Rating_num', color_discrete_sequence=['#E23744'],
            title='Customer rating distribution', labels={'Rating_num': 'Rating'}), use_container_width=True)
        monthly = filtered.dropna(subset=['Time']).assign(Month=lambda x: x['Time'].dt.to_period('M').astype(str)).groupby('Month').size().reset_index(name='Reviews')
        st.plotly_chart(px.line(monthly, x='Month', y='Reviews', title='Review activity over time', markers=True,
            color_discrete_sequence=['#E23744']), use_container_width=True)
        st.caption('Dataset dates: May 2016–May 2019. These charts show the provided sample, not current Zomato activity.')
with explorer:
    st.subheader('Compare restaurants')
    minimum = st.number_input('Minimum reviews per restaurant', min_value=1, max_value=1000, value=20)
    summary = filtered.groupby('Restaurant').agg(Reviews=('Rating_num', 'size'), Rating=('Rating_num', 'mean'),
        Cost=('Cost_num', 'first'), Cuisine=('Primary_Cuisine', 'first')).reset_index()
    summary = summary[summary['Reviews'] >= minimum].sort_values('Rating', ascending=False)
    if summary.empty:
        st.info('No restaurants match. Reduce the minimum reviews or widen the filters.')
    else:
        st.plotly_chart(px.bar(summary.head(15).sort_values('Rating'), x='Rating', y='Restaurant', orientation='h',
            title='Highest-rated restaurants in this selection', range_x=[0, 5], color_discrete_sequence=['#E23744']), use_container_width=True)
        st.dataframe(summary, hide_index=True, width='stretch',
            column_config={'Rating': st.column_config.NumberColumn(format='%.2f'), 'Cost': st.column_config.NumberColumn('Approx. cost for two (₹)', format='₹%d')})
        st.download_button('Download restaurant summary', summary.to_csv(index=False), 'restaurant_summary.csv', 'text/csv')
with predict:
    st.subheader('Try your own restaurant review')
    st.write('Enter an English review to predict Positive, Neutral, or Negative sentiment.')
    st.caption('Examples: “Excellent food and friendly service” • “The meal was average” • “Cold food and terrible service”')
    with st.form('review_form'):
        review = st.text_area('Your review', max_chars=5000, height=130, placeholder='How was the food, service, or ambience?')
        submitted = st.form_submit_button('Analyze sentiment', type='primary')
    if submitted:
        if not normalize_review(review):
            st.warning('Please enter a review containing meaningful words.')
        else:
            try:
                model = load_model()
                vector = model.named_steps['tfidf'].transform([review])
                if vector.nnz == 0:
                    st.warning('These words are outside the model vocabulary. Try a more descriptive English review.')
                else:
                    label = model.predict([review])[0]
                    st.success(f'Predicted sentiment: {label}')
                    probabilities = pd.DataFrame({'Sentiment': model.classes_, 'Model probability': model.predict_proba([review])[0]})
                    st.plotly_chart(px.bar(probabilities, x='Sentiment', y='Model probability', color='Sentiment',
                        color_discrete_map=COLORS, range_y=[0, 1]), use_container_width=True)
                    st.caption('Probabilities are model estimates, not guaranteed confidence. Mixed opinions, sarcasm, and unfamiliar language may be misclassified.')
            except (OSError, ValueError, ImportError) as exc:
                st.error(f'The prediction model is unavailable. Please check the deployment: {exc}')
with evaluation:
    st.subheader('How the model works')
    st.write('English review → shared text cleaning (negation preserved) → TF-IDF word and phrase features → Logistic Regression.')
    st.write('Labels come from customer ratings: 1–2 = Negative, above 2–3.5 = Neutral, above 3.5 = Positive. These are rating-derived labels, not independently annotated sentiment.')
    try:
        results = load_results()
        a, b, c = st.columns(3)
        a.metric('Held-out accuracy', f"{results['accuracy']:.2%}")
        b.metric('Macro F1', f"{results['f1_macro']:.2%}")
        c.metric('Held-out reviews', f"{results['test_rows']:,}")
        st.caption('Metrics below belong to the app model. The notebooks retain their original experiment results.')
        report = pd.DataFrame(results['classification_report']).T.loc[results['labels'], ['precision', 'recall', 'f1-score', 'support']]
        st.dataframe(report, width='stretch')
        st.plotly_chart(px.imshow(results['confusion_matrix'], x=results['labels'], y=results['labels'],
            text_auto=True, color_continuous_scale='Reds', labels={'x': 'Predicted', 'y': 'Actual'}, title='Held-out confusion matrix'), use_container_width=True)
        st.write(f"Training reviews: {results['train_rows']:,}. Identical normalized reviews are kept in the same split, including during cross-validation. Parameters are chosen using training-fold macro F1.")
        st.write('Limitations: historical restaurant sample; class imbalance; rating-derived labels; limited English vocabulary. Similar reviews and restaurant-specific patterns may still limit generalization.')
        with st.expander('Original notebook model comparison'):
            original = json.loads((ROOT / 'data/ml_model_results.json').read_text())
            st.dataframe(pd.DataFrame([{'Model': name, 'Accuracy': item['tuned_metrics']['accuracy'], 'Macro F1': item['tuned_metrics']['f1_macro']} for name, item in original['models'].items()]), hide_index=True)
            st.caption('Historical experiments used different preprocessing and a random row split; results are not directly comparable to the app evaluation.')
    except (OSError, ValueError) as exc:
        st.error(f'Model evaluation results could not be loaded: {exc}')
