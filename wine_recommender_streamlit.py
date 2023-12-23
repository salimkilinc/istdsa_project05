import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
from urllib.parse import urlparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


st.set_page_config(
    layout="wide",
    page_title="Wine Recommender",
    page_icon=":🍾:",
    menu_items={
        "Get help": "mailto:salimkilinc@yahoo.com",
        "About": "For More Information\n" + "https://github.com/salimkilinc"
    }
)

background_image = """
<style>
    .stApp {
        background-image: url('https://lh3.googleusercontent.com/pw/ABLVV86CFJQCmdItQ-ymRNCgBxuIPfCcJpjCKdkqkNPajsyzJqCHgjddUhZxtg2FuU7jVbOPcCfB16lNaXtqBiUSQdyOd0wtJeMoFeJz7KXFDaXPivSIEQnbXXjuexoTqC8TuTgbSgZUBmguypz5RoF407g2=w2466-h1518-s-no');
        background-size: cover;
        background-repeat: no-repeat;
    }
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

cosine_sim, df = joblib.load('content_based_recommender_model.pkl')
cosine_sim2_df, dist_euc_df, df_2 = joblib.load('collaborative_recommender_model.pkl')


def content_based_recommendations(WineName):
    matching_wines = df[df['WineName'].str.contains(WineName, case=False)]
    
    if matching_wines.empty:
        st.warning(f"No wines found for the given input: {WineName}")
        return pd.DataFrame(columns=['WineName', 'Website'])

    idx = matching_wines.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    wine_indices = [i[0] for i in sim_scores]
    content_based_output = df.loc[wine_indices, ['WineName', 'Website']].copy()
    
    return content_based_output

def collaborative_filtering_recommendations(wine_name, count=10):
    wine_name = wine_name.lower()
    
    wines_cosine = [wine for wine in df_2['WineName'] if wine.lower().startswith(wine_name)]
    wines_summed_cosine = cosine_sim2_df[wines_cosine].apply(lambda row: np.sum(row), axis=1)
    wines_summed_cosine = wines_summed_cosine.sort_values(ascending=False)
    ranked_wines_cosine = wines_summed_cosine.index[wines_summed_cosine.index.isin(wines_cosine) == False]
    ranked_wines_cosine = ranked_wines_cosine.tolist()
    ranked_wines_cosine = ranked_wines_cosine[:count]
    
    wines_euclidean = [wine for wine in df_2['WineName'] if wine.lower().startswith(wine_name)]
    wines_summed_euclidean = dist_euc_df[wines_euclidean].apply(lambda row: np.sum(row), axis=1)
    wines_summed_euclidean = wines_summed_euclidean.sort_values()
    ranked_wines_euclidean = wines_summed_euclidean.index[wines_summed_euclidean.index.isin(wines_euclidean) == False]
    ranked_wines_euclidean = ranked_wines_euclidean.tolist()
    ranked_wines_euclidean = ranked_wines_euclidean[:count]

    combined_ranking = {}
    for i, wine in enumerate(ranked_wines_cosine):
        combined_ranking[wine] = combined_ranking.get(wine, 0) + i

    for i, wine in enumerate(ranked_wines_euclidean):
        combined_ranking[wine] = combined_ranking.get(wine, 0) + i

    sorted_combined_ranking = sorted(combined_ranking.items(), key=lambda x: x[1])

    top_10_wine_indices = [df_2.index[df_2['WineName'] == wine].tolist()[0] for wine, _ in sorted_combined_ranking[:count]]
    top_10_wines = df_2.loc[top_10_wine_indices, ['WineName', 'Website']].copy()
    
    return top_10_wines


def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def main():
    st.title("Wine Recommender")
    user_input = st.text_input("Type the initial letters of a wine name and hit Enter:")
    st.markdown("---")

    if user_input:
        matching_wines = df_2[df_2['WineName'].str.lower().str.contains(user_input.lower())]
        unique_wine_names = matching_wines['WineName'].unique()
        selected_wine = st.selectbox("Select a Wine:", [' --Select a Wine--'] + list(unique_wine_names), index=0)

        if selected_wine != ' --Select a Wine--':
            content_based_output = content_based_recommendations(selected_wine)
            collaborative_output = collaborative_filtering_recommendations(selected_wine, count=10)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Related Wines:")
                content_based_output['Website'] = content_based_output['Website'].apply(lambda x: f'https://{x}')
                content_based_output['Website'] = content_based_output['Website'].apply(lambda x: f'<a href="{x}" target="_blank">{extract_domain(x)}</a>')
                content_based_html = content_based_output.to_html(escape=False, index=False)
                st.markdown(content_based_html, unsafe_allow_html=True)
            with col2:
                st.subheader("People Also Liked:")
                collaborative_output['Website'] = collaborative_output['Website'].apply(lambda x: f'https://{x}')
                collaborative_output['Website'] = collaborative_output['Website'].apply(lambda x: f'<a href="{x}" target="_blank">{extract_domain(x)}</a>')
                collaborative_html = collaborative_output.to_html(escape=False, index=False)
                st.markdown(collaborative_html, unsafe_allow_html=True)
            
            st.markdown("---")
            st.write("This project utilizes the [X-Wines](%s) dataset as a key data source." % "https://www.mdpi.com/2504-2289/7/1/20")
            st.markdown("---")

if __name__ == "__main__":
    main()
