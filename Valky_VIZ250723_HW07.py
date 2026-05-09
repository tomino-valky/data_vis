import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="Horse Colic Dashboard",
    page_icon="🐴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* 1. Set the background image for the entire app */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 2. Add a DARK semi-transparent overlay so white text is readable */
.block-container {
    background-color: rgba(14, 17, 23, 0.85); /* Streamlit dark mode color, 85% opacity */
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.5); /* Slightly darker shadow for depth */
    margin-top: 2rem;
}

/* 3. Your existing custom headers and dataframe styles */
.main-header {
    font-size: 2.2rem;
    color: #4A90E2;
    font-weight: 700;
}
.sub-header {
    font-size: 1.5rem;
    color: #E0E0E0; /* Changed from dark gray to light gray for readability */
    font-style: italic;
}
.stDataFrame {
    border: 1px solid #444444; /* Darkened the border for dark mode */
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Data Loading
# ----------------------------------------------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'horse.csv')
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    df['outcome'] = df['outcome'].fillna('Unknown').str.capitalize()
    df['surgery'] = df['surgery'].fillna('Unknown').str.capitalize()
    df['age'] = df['age'].fillna('Unknown').str.capitalize()
    
    if 'mucous_membrane' in df.columns:
        df['mucous_membrane'] = df['mucous_membrane'].fillna('Unknown').str.replace('_', ' ').str.title()
        
    return df

raw_df = load_data()

# ----------------------------------------------------
# 3. Sidebar Filters
# ----------------------------------------------------
st.sidebar.title("🛠️ Filter Controls")
st.sidebar.markdown("Filter the clinical data to dynamically update all views.")

# Age
all_ages = list(raw_df['age'].unique())
selected_ages = st.sidebar.multiselect("Select Age Group:", options=all_ages, default=all_ages)

# Surgery
all_surgeries = list(raw_df['surgery'].unique())
selected_surgeries = st.sidebar.multiselect("Had Surgery?:", options=all_surgeries, default=all_surgeries)

# Outcome
all_outcomes = list(raw_df['outcome'].unique())
selected_outcomes = st.sidebar.multiselect("Outcome:", options=all_outcomes, default=[o for o in all_outcomes if o != 'Unknown'])

# Pulse Slider
min_pulse = float(raw_df['pulse'].min() if pd.notnull(raw_df['pulse'].min()) else 0)
max_pulse = float(raw_df['pulse'].max() if pd.notnull(raw_df['pulse'].max()) else 200)
pulse_range = st.sidebar.slider("Pulse Range (BPM):", min_value=min_pulse, max_value=max_pulse, value=(min_pulse, max_pulse))

# ----------------------------------------------------
# 4. Apply Filters
# ----------------------------------------------------
filtered_df = raw_df[
    (raw_df['age'].isin(selected_ages)) &
    (raw_df['surgery'].isin(selected_surgeries)) &
    (raw_df['outcome'].isin(selected_outcomes)) &
    (raw_df['pulse'].fillna(pulse_range[0]) >= pulse_range[0]) &
    (raw_df['pulse'].fillna(pulse_range[1]) <= pulse_range[1])
]

# ----------------------------------------------------
# 5. Header & Description
# ----------------------------------------------------
st.markdown('<p class="main-header">🐴 Horse Colic Exploratory Dashboard</p>', unsafe_allow_html=True)
st.markdown(f"**Dataset Size (Filtered):** {filtered_df.shape[0]} records | **Total Features:** {filtered_df.shape[1]}")
st.markdown("""
**Introduction:**
This interactive dashboard visualizes clinical parameters from the Horse Colic dataset. The goal is to explore how variables like pulse, rectal temperature, and mucous membrane relate to the primary surgical outcomes. Use the controls on the left to drill down into specific subgroups. All charts and calculations dynamically update to reflect the subset chosen!
""")

# Show Data Table
with st.expander("🔍 View Filtered Data Table"):
    st.dataframe(filtered_df, width="stretch")

if filtered_df.shape[0] < 5:
    st.warning("Not enough data to generate robust visualizations! Please loosen the filter restrictions.")
    st.stop()

st.divider()

# ----------------------------------------------------
# 6. Generate Master Analysis Metrics (PCA & t-SNE precalc)
# ----------------------------------------------------
# Select numeric features for multidimension plots
numeric_cols = ['pulse', 'rectal_temp', 'respiratory_rate', 'total_protein', 'packed_cell_volume']
# Filter out rows completely missing target or we just dropnas on subset
df_ml = filtered_df.dropna(subset=numeric_cols, how='all').copy()

if df_ml.shape[0] > 5:
    X = df_ml[numeric_cols]
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Calculate PCA
    pca = PCA(n_components=min(5, X_scaled.shape[0], X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    df_ml['PC1'] = X_pca[:, 0]
    df_ml['PC2'] = X_pca[:, 1]
    explained_var = pca.explained_variance_ratio_

    # Calculate t-SNE
    perplexity_val = min(30, df_ml.shape[0] - 1)
    if perplexity_val > 0:
        tsne = TSNE(n_components=2, perplexity=perplexity_val, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        df_ml['TSNE1'] = X_tsne[:, 0]
        df_ml['TSNE2'] = X_tsne[:, 1]
else:
    df_ml = pd.DataFrame()


# ----------------------------------------------------
# 7. Rendering the 6 Plots
# ----------------------------------------------------
color_map = {'Lived': '#388E3C', 'Died': '#D32F2F', 'Euthanized': '#9E9E9E', 'Unknown': '#555555'}

tab1, tab2, tab3 = st.tabs(["📊 Standard Metrics", "🧬 Dimensionality Projections (PCA & t-SNE)", "🌐 Parallel Coordinates"])

with tab1:
    col1, col2 = st.columns(2)
    
    # Plot 1: Pulse by Outcome
    with col1:
        st.subheader("1. Average Pulse by Outcome")
        st.markdown("Shows average heart rate broken down by patient outcome. This metric updates automatically if you filter out ages, or restrict the pulse slider.")
        avg_pulse = filtered_df.groupby('outcome', as_index=False)['pulse'].mean()
        fig1 = px.bar(avg_pulse, x='outcome', y='pulse', text_auto='.1f',
                      color='outcome', color_discrete_map=color_map,
                      labels={'outcome': 'Outcome', 'pulse': 'Avg Pulse (bpm)'})
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, width="stretch")
        
    # Plot 2: Rectal Temp by Mucous Membrane
    with col2:
        st.subheader("2. Rectal Temp by Mucous Membrane")
        st.markdown("Compares mean core temperature across clinical mucous membrane classifications.")
        if 'mucous_membrane' in filtered_df.columns:
            avg_temp = filtered_df.groupby('mucous_membrane', as_index=False)['rectal_temp'].mean()
            fig2 = px.bar(avg_temp, x='mucous_membrane', y='rectal_temp', 
                          color='mucous_membrane', color_discrete_sequence=px.colors.qualitative.Pastel,
                          labels={'mucous_membrane': 'Membrane Type', 'rectal_temp': 'Avg Rectal Temp (°C)'})
            fig2.update_layout(showlegend=False)
            fig2.update_yaxes(range=[35, int(filtered_df['rectal_temp'].max()+2) if pd.notnull(filtered_df['rectal_temp'].max()) else 40])
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("Mucous Membrane data unavailable.")

with tab2:
    if not df_ml.empty:
        col3, col4 = st.columns(2)
        
        # Plot 4: PCA Plot
        with col3:
            st.subheader("3. PCA Projection (2D)")
            st.markdown("Advanced linear multivariate view. Plots PC1 vs PC2 computed actively on the filtered dataset.")
            fig_pca = px.scatter(df_ml, x='PC1', y='PC2', color='outcome', color_discrete_map=color_map,
                                 hover_data=['pulse', 'rectal_temp', 'age'], size_max=10)
            fig_pca.update_layout(legend_title_text='Outcome')
            st.plotly_chart(fig_pca, width="stretch")

        # Plot 5: t-SNE Plot
        with col4:
            st.subheader("4. t-SNE Projection")
            st.markdown("Advanced non-linear manifold mapping algorithm designed to cluster similar filtered instances tightly.")
            if 'TSNE1' in df_ml.columns:
                fig_tsne = px.scatter(df_ml, x='TSNE1', y='TSNE2', color='outcome', color_discrete_map=color_map,
                                     hover_data=['pulse', 'rectal_temp', 'age'], size_max=10)
                fig_tsne.update_layout(legend_title_text='Outcome')
                st.plotly_chart(fig_tsne, width="stretch")
            else:
                st.info("Not enough data to calculate t-SNE.")
        
        st.divider()
        # Plot 6: Scree Plot
        st.subheader("5. PCA Components Variance (Scree Plot)")
        st.markdown("Details exactly how much variance the active calculated PCA components capture. The line represents the cumulative sum.")
        
        fig_scree = go.Figure()
        components_x = np.arange(1, len(explained_var) + 1)
        
        fig_scree.add_trace(go.Bar(
            x=components_x, y=explained_var * 100,
            name='Individual Variance', marker_color='#4A90E2'
        ))
        fig_scree.add_trace(go.Scatter(
            x=components_x, y=np.cumsum(explained_var) * 100,
            mode='lines+markers', name='Cumulative Variance', line=dict(color='#E94B3C', width=3)
        ))
        fig_scree.update_layout(
            xaxis=dict(title='Principal Component', tickmode='linear'),
            yaxis=dict(title='Variance Explained (%)'),
            legend=dict(x=0.01, y=0.99),
            hovermode="x unified"
        )
        st.plotly_chart(fig_scree, width="stretch")
    else:
        st.info("Insufficient analytical data for Dimensionality Reduction.")

with tab3:
    st.subheader("6. Multivariate Parallel Coordinates")
    st.markdown("Drag over the vertical axes to highlight/brush specific ranges. The color tracks the patient's 'Outcome'. Notice how overlapping lines reveal physiological confusion.")
    
    if not df_ml.empty:
        # Convert ordinal outcome to numeric for Plotly Parallel Coordinates
        outcome_mapping = {'Lived': 0, 'Euthanized': 1, 'Died': 2, 'Unknown': 3}
        df_ml['outcome_num'] = df_ml['outcome'].map(outcome_mapping)
        
        fig_par = go.Figure(data=
            go.Parcoords(
                # Set fonts to light grey so they are visible on the dark background
                labelfont=dict(color='#E0E0E0', size=13),
                tickfont=dict(color='#E0E0E0'),
                rangefont=dict(color='#E0E0E0'),
                line=dict(color=df_ml['outcome_num'],
                          colorscale=[[0, '#388E3C'], [0.5, '#9E9E9E'], [1, '#D32F2F']],
                          showscale=True, 
                          colorbar=dict(
                              title=dict(
                                  text='Outcome',
                                  font=dict(color='#E0E0E0')
                              ),
                              tickvals=[0,1,2], 
                              ticktext=['Lived', 'Euthanized', 'Died'],
                              tickfont=dict(color='#E0E0E0')
                          )
                         ),
                # Matches the order and names from your previous assignment
                dimensions=list([
                    dict(range=[df_ml['rectal_temp'].min(), df_ml['rectal_temp'].max()], label='Rectal Temp (°C)', values=df_ml['rectal_temp']),
                    dict(range=[df_ml['pulse'].min(), df_ml['pulse'].max()], label='Pulse (bpm)', values=df_ml['pulse']),
                    dict(range=[df_ml['respiratory_rate'].min(), df_ml['respiratory_rate'].max()], label='Respiratory Rate', values=df_ml['respiratory_rate']),
                    dict(range=[df_ml['packed_cell_volume'].min(), df_ml['packed_cell_volume'].max()], label='Packed Cell Vol (%)', values=df_ml['packed_cell_volume']),
                    dict(range=[df_ml['total_protein'].min(), df_ml['total_protein'].max()], label='Total Protein (g)', values=df_ml['total_protein'])
                ])
            )
        )
        
        # Add margins to prevent labels from being cut off, and make background transparent
        fig_par.update_layout(
            margin=dict(l=80, r=80, t=60, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_par, width="stretch")
    else:
        st.info("Not enough data to calculate parallel coordinates.")