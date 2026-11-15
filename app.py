import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression

# Page settings
st.set_page_config(
    page_title="Car Data Visualization",
    layout="wide"
)

# Load dataset
df = pd.read_csv("car_data.csv", sep=";")


# ==========================================
# DATA PREPROCESSING - OUTLIER DETECTION
# IQR (Interquartile Range) Method
# ==========================================

st.subheader("Data Preprocessing - Outlier Detection")

st.write(
    "Potential outliers are detected using the IQR method. "
    "Valid extreme car values are retained for analysis."
)

outlier_columns = ["MSRP", "Engine HP", "city mpg"]

outlier_results = []

for column in outlier_columns:

    # Convert values to numeric
    values = pd.to_numeric(df[column], errors="coerce").dropna()

    # Calculate IQR
    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)
    IQR = Q3 - Q1

    # Determine outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outlier_count = (
        (values < lower_bound) |
        (values > upper_bound)
    ).sum()

    outlier_results.append({
        "Feature": column,
        "Lower Bound": round(lower_bound, 2),
        "Upper Bound": round(upper_bound, 2),
        "Outlier Count": int(outlier_count)
    })

# Show results in an expandable section
with st.expander("Data Quality Check - IQR Outlier Detection"):

    st.dataframe(
        pd.DataFrame(outlier_results),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Outliers are identified but not automatically removed. "
        "High prices and horsepower may represent genuine "
        "luxury or performance vehicles."
    )


# Data preprocessing
df = df.dropna(subset=[
    "Engine Fuel Type",
    "Engine HP",
    "Engine Cylinders"
])

# ==========================================
# FUEL CATEGORY PREPROCESSING
# Required for Graph 9 - Logistic Regression
# ==========================================

def categorize_fuel(fuel):
    f = str(fuel).lower()

    if "premium" in f:
        return "Premium Fuel"
    elif "regular" in f:
        return "Regular Fuel"

    return "Other"


df["Fuel Category"] = df["Engine Fuel Type"].apply(categorize_fuel)

# ==========================================
# SYNCHRONIZED BRAND FILTERS (ALL 9 GRAPHS)
# ==========================================
all_brands = sorted(df["Make"].dropna().unique().tolist())

brand_widget_keys = [
    "heatmap_brands", "parallel_brands",
    "bubble_brands", "sunburst_brands",
    "cluster_brands", "dendro_brands",
    "sankey_brands", "g8_brands"
]

if "shared_brands" not in st.session_state:
    st.session_state["shared_brands"] = [
        b for b in ["BMW", "Audi", "Toyota", "Ford"] if b in all_brands
    ]

# Initialize widget state BEFORE creating any widgets.
for widget_key in brand_widget_keys:
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state["shared_brands"].copy()

if "surface_brand" not in st.session_state:
    st.session_state["surface_brand"] = (
        st.session_state["shared_brands"][0]
        if st.session_state["shared_brands"] else None
    )

def synchronize_brands(source_key):
    """Any brand multiselect updates every other brand selector."""
    chosen = list(st.session_state[source_key])
    st.session_state["shared_brands"] = chosen.copy()
    for widget_key in brand_widget_keys:
        if widget_key != source_key:
            st.session_state[widget_key] = chosen.copy()
    # The surface uses one brand, so keep it within the shared selection.
    if st.session_state.get("surface_brand") not in chosen:
        st.session_state["surface_brand"] = chosen[0] if chosen else None

def synchronize_surface_brand():
    """Choosing a brand in Graph 9 filters all other charts to that brand."""
    brand = st.session_state["surface_brand"]
    chosen = [brand] if brand else []
    st.session_state["shared_brands"] = chosen.copy()
    for widget_key in brand_widget_keys:
        st.session_state[widget_key] = chosen.copy()



# ====================================
# GRAPH 1 - ADVANCED HEATMAP
# Fuel Economy Comparison by Brand
# Year Filter, Dynamic Sorting and Analysis
# ====================================

st.header("Graph 1 - Heatmap: Fuel Economy by Brand")

st.write(
    "Compare average city and highway fuel consumption "
    "across car brands. Filter by production year, "
    "sort brands and identify the most fuel-efficient cars."
)

# Select car brands (synchronized with other graphs)
heatmap_brands = st.multiselect(
    "Select brands for Heatmap:",
    all_brands,
    key="heatmap_brands",
    on_change=synchronize_brands,
    args=("heatmap_brands",)
)

# Prepare data
heatmap_df = df[df["Make"].isin(heatmap_brands)].copy()

for column in ["Year", "city mpg", "highway MPG"]:
    heatmap_df[column] = pd.to_numeric(
        heatmap_df[column], errors="coerce"
    )

heatmap_df = heatmap_df.dropna(
    subset=["Year", "city mpg", "highway MPG"]
)

heatmap_df = heatmap_df[
    (heatmap_df["city mpg"] > 0) &
    (heatmap_df["highway MPG"] > 0)
].copy()

# ====================================
# YEAR FILTER
# ====================================

available_years = sorted(
    pd.to_numeric(df["Year"], errors="coerce")
    .dropna().astype(int).unique().tolist()
)

if available_years:

    year_range = st.slider(
        "Select Production Year Range:",
        min_value=min(available_years),
        max_value=max(available_years),
        value=(min(available_years), max(available_years)),
        key="heatmap_year_range"
    )

    heatmap_df = heatmap_df[
        heatmap_df["Year"].between(
            year_range[0], year_range[1]
        )
    ]

# ====================================
# DYNAMIC SORTING
# ====================================

sort_option = st.selectbox(
    "Sort Brands By:",
    [
        "Alphabetical",
        "Lowest Fuel Consumption",
        "Highest Fuel Consumption"
    ],
    key="heatmap_sort"
)

if not heatmap_df.empty:

    # Convert MPG to L/100 km
    heatmap_df["City"] = (
        235.215 / heatmap_df["city mpg"]
    )

    heatmap_df["Highway"] = (
        235.215 / heatmap_df["highway MPG"]
    )

    # Calculate average consumption
    heatmap_data = (
        heatmap_df.groupby("Make")[
            ["City", "Highway"]
        ]
        .mean()
    )

    # Combined average consumption for sorting
    heatmap_data["Average"] = (
        heatmap_data["City"] +
        heatmap_data["Highway"]
    ) / 2

    # Sort brands dynamically
    if sort_option == "Lowest Fuel Consumption":
        heatmap_data = heatmap_data.sort_values(
            "Average", ascending=True
        )

    elif sort_option == "Highest Fuel Consumption":
        heatmap_data = heatmap_data.sort_values(
            "Average", ascending=False
        )

    else:
        heatmap_data = heatmap_data.sort_index()

    # ====================================
    # CREATE HEATMAP
    # ====================================

    fig2 = px.imshow(
        heatmap_data[["City", "Highway"]].round(2),
        labels={
            "x": "Driving Condition",
            "y": "Car Brand",
            "color": "L/100 km"
        },
        x=["City", "Highway"],
        y=heatmap_data.index,
        color_continuous_scale="RdYlGn_r",
        text_auto=True,
        aspect="auto",
        title="Average Fuel Consumption (L/100 km)"
    )

    fig2.update_layout(
        height=max(450, len(heatmap_data) * 45)
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # ====================================
    # AUTOMATIC ANALYSIS
    # ====================================

    st.subheader("Automatic Fuel Efficiency Analysis")

    # Identify best and worst brands
    best_brand = heatmap_data["Average"].idxmin()
    worst_brand = heatmap_data["Average"].idxmax()

    best_value = heatmap_data.loc[
        best_brand, "Average"
    ]

    worst_value = heatmap_data.loc[
        worst_brand, "Average"
    ]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Most Fuel-Efficient Brand",
            value=best_brand,
            delta=f"{best_value:.2f} L/100 km",
            delta_color="off"
        )

    with col2:
        st.metric(
            label="Highest Fuel Consumption Brand",
            value=worst_brand,
            delta=f"{worst_value:.2f} L/100 km",
            delta_color="off"
        )

    st.info(
        "The analysis automatically updates when "
        "brands or production years are changed. "
        "Results are based on the average of city "
        "and highway fuel consumption."
    )

else:
    st.warning(
        "No cars match the selected brands "
        "and production year range."
    )





# ====================================
# GRAPH 2 - PARALLEL COORDINATES
# Price, Fuel Economy and Horsepower
# ====================================

st.header("Graph 2 - Parallel Coordinates: Car Comparison")

st.write(
    "Compare car prices, city fuel consumption and "
    "engine horsepower to explore affordable, "
    "fuel-efficient and powerful cars."
)

# Maximum car price
max_price = st.slider(
    "Maximum Car Price (USD):",
    min_value=10000,
    max_value=200000,
    value=30000,
    step=5000,
    key="graph3_budget"
)

# Select car brands
parallel_brands = st.multiselect(
    "Select brands for performance comparison:",
    all_brands,
    key="parallel_brands",
    on_change=synchronize_brands,
    args=("parallel_brands",)
)

# Filter by price and selected brands
parallel_df = df[
    (df["MSRP"] <= max_price) &
    (df["Make"].isin(parallel_brands))
].copy()

# Convert columns to numeric
for column in ["MSRP", "Engine HP", "city mpg"]:
    parallel_df[column] = pd.to_numeric(
        parallel_df[column], errors="coerce"
    )

# Remove invalid data
parallel_df = parallel_df.dropna(
    subset=["MSRP", "Engine HP", "city mpg"]
)

parallel_df = parallel_df[
    (parallel_df["MSRP"] > 0) &
    (parallel_df["Engine HP"] > 0) &
    (parallel_df["city mpg"] > 0)
]

if not parallel_df.empty:

    # Convert MPG to L/100 km
    parallel_df["Fuel Consumption"] = (
        235.215 / parallel_df["city mpg"]
    )

    # Create parallel coordinates
    fig3 = px.parallel_coordinates(
        parallel_df,
        dimensions=[
            "MSRP",
            "Fuel Consumption",
            "Engine HP"
        ],
        color="Engine HP",
        labels={
            "MSRP": "Price (USD)",
            "Fuel Consumption": "City Fuel (L/100 km)",
            "Engine HP": "Horsepower (HP)"
        },
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Price, Fuel Consumption and Horsepower"
    )

   
    fig3.update_traces(
        dimensions=[
            dict(
                label="Price (USD)",
                values=parallel_df["MSRP"],
                range=[0, max_price],
                tickvals=[
                    0,
                    max_price * 0.25,
                    max_price * 0.50,
                    max_price * 0.75,
                    max_price
                ],
                ticktext=[
                    f"${int(v):,}"
                    for v in [
                        0,
                        max_price * 0.25,
                        max_price * 0.50,
                        max_price * 0.75,
                        max_price
                    ]
                ]
            ),
            dict(
                label="City Fuel (L/100 km)",
                values=parallel_df["Fuel Consumption"]
            ),
            dict(
                label="Horsepower (HP)",
                values=parallel_df["Engine HP"]
            )
        ]
    )

    fig3.update_layout(
        height=550,
        margin=dict(l=110, r=80, t=80, b=60)
    )


    st.plotly_chart(fig3, use_container_width=True)

    

    st.write(
        "Number of matching car records:",
        len(parallel_df)
    )

else:
    st.warning(
        "No cars match your selected budget and brands."
    )

# ==========================================
# GRAPH 3 - K-MEANS CLUSTERING (ML 1)
# 3D Car Clustering
# ==========================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.header("Graph 3 - 3D Car Clustering (K-Means)")

st.write(
    "Discover groups of similar cars based on "
    "price, city fuel consumption and horsepower."
)

# Select car brands
cluster_brands = st.multiselect(
    "Select brands for clustering:",
    all_brands,
    key="cluster_brands",
    on_change=synchronize_brands,
    args=("cluster_brands",)
)

# Select maximum budget
cluster_budget = st.slider(
    "Maximum Car Price (USD) - Clustering:",
    min_value=10000,
    max_value=500000,
    value=100000,
    step=5000,
    key="cluster_budget"
)

# Select number of clusters
cluster_count = st.slider(
    "Number of clusters:",
    min_value=2,
    max_value=6,
    value=3,
    step=1,
    key="cluster_count"
)

# Prepare data
cluster_df = df.copy()

for column in ["MSRP", "Engine HP", "city mpg"]:
    cluster_df[column] = pd.to_numeric(
        cluster_df[column],
        errors="coerce"
    )

cluster_df = cluster_df.dropna(
    subset=["Make", "Model", "Year",
            "MSRP", "Engine HP", "city mpg"]
)

cluster_df = cluster_df[
    (cluster_df["Make"].isin(cluster_brands)) &
    (cluster_df["MSRP"] > 0) &
    (cluster_df["MSRP"] <= cluster_budget) &
    (cluster_df["Engine HP"] > 0) &
    (cluster_df["city mpg"] > 0)
].copy()

if len(cluster_df) >= cluster_count:

    # Convert MPG to L/100 km
    cluster_df["Fuel Consumption"] = (
        235.215 / cluster_df["city mpg"]
    )

    # Features used by the ML algorithm
    features = [
        "MSRP",
        "Fuel Consumption",
        "Engine HP"
    ]

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(
        cluster_df[features]
    )

    # Apply K-Means
    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )

    cluster_df["Cluster"] = kmeans.fit_predict(
        scaled_features
    ).astype(str)

    # Create readable car names
    cluster_df["Car"] = (
        cluster_df["Make"].astype(str) + " " +
        cluster_df["Model"].astype(str) + " (" +
        cluster_df["Year"].astype(str) + ")"
    )

    # Create interactive 3D chart
    fig6 = px.scatter_3d(
        cluster_df,
        x="MSRP",
        y="Fuel Consumption",
        z="Engine HP",
        color="Cluster",
        hover_name="Car",
        hover_data={
            "MSRP": ":,.0f",
            "Fuel Consumption": ":.2f",
            "Engine HP": ":.0f",
            "Cluster": True
        },
        labels={
            "MSRP": "Price (USD)",
            "Fuel Consumption": "City Fuel (L/100 km)",
            "Engine HP": "Horsepower (HP)"
        },
        title="K-Means Car Clusters",
        opacity=0.75
    )

    fig6.update_traces(
        marker=dict(size=4)
    )

    fig6.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        legend_title_text="ML Cluster"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    

else:
    st.warning(
        "Not enough cars match your selections "
        "to create the requested number of clusters."
    )





# ====================================
# GRAPH 4 - BUBBLE CHART
# Car Model Comparison
# ====================================

st.header("Graph 4 - Bubble Chart: Car Model Comparison")

st.write(
    "Select car brands to compare their models "
    "based on price, horsepower and fuel consumption."
)

# Select brands
bubble_brands = st.multiselect(
    "Select brands for model comparison:",
    all_brands,
    key="bubble_brands",
    on_change=synchronize_brands,
    args=("bubble_brands",)
)

# Filter selected brands
bubble_df = df[df["Make"].isin(bubble_brands)].copy()

# Convert columns to numeric
for col in ["MSRP", "Engine HP", "city mpg"]:
    bubble_df[col] = pd.to_numeric(
        bubble_df[col], errors="coerce"
    )

# Remove missing and invalid values
bubble_df = bubble_df.dropna(
    subset=["MSRP", "Engine HP", "city mpg"]
)

bubble_df = bubble_df[
    (bubble_df["MSRP"] > 0) &
    (bubble_df["Engine HP"] > 0) &
    (bubble_df["city mpg"] > 0)
]

if not bubble_df.empty:

    # Convert MPG to L/100 km
    bubble_df["Fuel Consumption"] = (
        235.215 / bubble_df["city mpg"]
    )

   
    # Group different years under each model
    model_data = bubble_df.groupby(
        ["Make", "Model"],
        as_index=False
    ).agg(
        Price=("MSRP", "mean"),
        Horsepower=("Engine HP", "mean"),
        Fuel=("Fuel Consumption", "mean")
    )

    
    # User-selected maximum budget
    bubble_max_price = st.slider(
        "Maximum Model Price (USD):",
        min_value=10000,
        max_value=500000,
        value=100000,
        step=5000,
        key="bubble_budget"
    )

    # Filter models by selected budget
    model_data = model_data[
        model_data["Price"] <= bubble_max_price
    ]


    # Select up to 5 affordable models per brand
    # Prioritize lower fuel consumption
    model_data = (
        model_data
        .sort_values(["Make", "Fuel", "Price"])
        .groupby("Make", group_keys=False)
        .head(5)
        .reset_index(drop=True)
    )

    # Create model labels
    model_data["Model Name"] = (
        model_data["Make"] + " " + model_data["Model"]
    )


   

    # Optional model selection
    selected_models = st.multiselect(
        "Select specific models (optional):",
        sorted(model_data["Model Name"].unique()),
        key="bubble_models"
    )

    if selected_models:
        model_data = model_data[
            model_data["Model Name"].isin(selected_models)
        ]

    # Create bubble chart
    fig4 = px.scatter(
        model_data,
        x="Price",
        y="Horsepower",
        size="Horsepower",
        color="Fuel",
        hover_name="Model Name",
        hover_data={
            "Price": ":,.0f",
            "Horsepower": ":.0f",
            "Fuel": ":.2f"
        },
        color_continuous_scale="RdYlGn_r",
        labels={
            "Price": "Average Price (USD)",
            "Horsepower": "Average Horsepower (HP)",
            "Fuel": "City Fuel Consumption (L/100 km)"
        },
        title="Car Model Price, Power and Fuel Comparison",
        size_max=30
    )

    fig4.update_layout(height=600)

    st.plotly_chart(fig4, use_container_width=True)

    st.write("Models displayed:", len(model_data))

else:
    st.info("Please select at least one brand with valid data.")



# ==========================================
# GRAPH 5 - SUNBURST CHART
# Brand, Model, Transmission and Drivetrain
# ==========================================

st.header("Graph 5 - Sunburst: Car Features Explorer")

st.write(
    "Explore car brands, models, transmission types "
    "and drivetrain systems by clicking the chart."
)

# Select brands
sunburst_brands = st.multiselect(
    "Select car brands:",
    all_brands,
    key="sunburst_brands",
    on_change=synchronize_brands,
    args=("sunburst_brands",)
)

# Select budget
sunburst_budget = st.slider(
    "Maximum Car Price (USD) - Sunburst:",
    min_value=10000,
    max_value=500000,
    value=100000,
    step=5000,
    key="sunburst_budget"
)

# Prepare data
sunburst_df = df.copy()

sunburst_df["MSRP"] = pd.to_numeric(
    sunburst_df["MSRP"],
    errors="coerce"
)

sunburst_df = sunburst_df[
    (sunburst_df["Make"].isin(sunburst_brands)) &
    (sunburst_df["MSRP"] > 0) &
    (sunburst_df["MSRP"] <= sunburst_budget)
].copy()

# Remove missing values
sunburst_df = sunburst_df.dropna(
    subset=[
        "Make",
        "Model",
        "Transmission Type",
        "Driven_Wheels"
    ]
)

# Remove unknown values
for column in ["Transmission Type", "Driven_Wheels"]:
    sunburst_df = sunburst_df[
        ~sunburst_df[column].astype(str).str.lower().isin(
            ["unknown", "nan", ""]
        )
    ]

if not sunburst_df.empty:

    # Group identical combinations
    sunburst_data = (
        sunburst_df.groupby(
            [
                "Make",
                "Model",
                "Transmission Type",
                "Driven_Wheels"
            ]
        )
        .agg(
            Car_Count=("MSRP", "size"),
            Average_Price=("MSRP", "mean")
        )
        .reset_index()
    )

    # Create interactive Sunburst Chart
    fig5 = px.sunburst(
        sunburst_data,
        path=[
            "Make",
            "Model",
            "Transmission Type",
            "Driven_Wheels"
        ],
        values="Car_Count",
        color="Average_Price",
        color_continuous_scale="RdYlGn_r",
        hover_data={
            "Average_Price": ":,.0f"
        },
        title="Brand → Model → Transmission → Drivetrain"
    )

    fig5.update_traces(
        textinfo="label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Car records: %{value}<br>"
            "Average price: $%{color:,.0f}"
            "<extra></extra>"
        )
    )

    
    fig5.update_layout(
        height=700,
        margin=dict(l=20, r=20, t=70, b=20),
        coloraxis=dict(
            cmin=0,
            cmax=sunburst_budget,
            colorbar=dict(
                title="Average Price (USD)",
                tickformat=",.0f"
            )
        )
    )


    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.info(
        "Click a segment to zoom into its details. "
        "Click the center to return to the previous level. "
        "Hover over a segment to see its record count "
        "and average price."
    )

else:
    st.warning(
        "No cars match your selected brands and budget."
    )



# ==========================================
# GRAPH 6 - HIERARCHICAL CLUSTERING
# Car Similarity Dendrogram
# ==========================================

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.header("Graph 6 - Car Similarity Dendrogram")

st.write(
    "Discover which car models are most similar based on "
    "price, horsepower and city fuel consumption."
)

# Select brands
dendro_brands = st.multiselect(
    "Select brands for similarity analysis:",
    all_brands,
    key="dendro_brands",
    on_change=synchronize_brands,
    args=("dendro_brands",)
)

# Select budget
dendro_budget = st.slider(
    "Maximum Car Price (USD) - Dendrogram:",
    min_value=10000,
    max_value=500000,
    value=100000,
    step=5000,
    key="dendro_budget"
)

# Number of models to compare
dendro_count = st.slider(
    "Number of car models to compare:",
    min_value=5,
    max_value=30,
    value=15,
    step=1,
    key="dendro_count"
)

# Prepare data
dendro_df = df.copy()

for col in ["MSRP", "Engine HP", "city mpg", "Year"]:
    dendro_df[col] = pd.to_numeric(
        dendro_df[col],
        errors="coerce"
    )

dendro_df = dendro_df.dropna(
    subset=["Make", "Model", "Year",
            "MSRP", "Engine HP", "city mpg"]
)

dendro_df = dendro_df[
    (dendro_df["Make"].isin(dendro_brands)) &
    (dendro_df["MSRP"] > 0) &
    (dendro_df["MSRP"] <= dendro_budget) &
    (dendro_df["Engine HP"] > 0) &
    (dendro_df["city mpg"] > 0)
].copy()

if not dendro_df.empty:

    # Convert fuel economy to consumption
    dendro_df["Fuel Consumption"] = (
        235.215 / dendro_df["city mpg"]
    )

    # Average characteristics per car model
    dendro_models = (
        dendro_df.groupby(
            ["Make", "Model"],
            as_index=False
        )
        .agg(
            Price=("MSRP", "mean"),
            Horsepower=("Engine HP", "mean"),
            Fuel=("Fuel Consumption", "mean")
        )
    )

    # Select representative models:
    # most affordable models from each selected brand
    dendro_models = (
        dendro_models.sort_values("Price")
        .groupby("Make", group_keys=False)
        .head(dendro_count)
        .sort_values("Price")
        .head(dendro_count)
        .reset_index(drop=True)
    )

    if len(dendro_models) >= 2:

        # Create car labels
        dendro_models["Car"] = (
            dendro_models["Make"] + " " +
            dendro_models["Model"]
        )

        # Standardize features
        feature_columns = [
            "Price",
            "Horsepower",
            "Fuel"
        ]

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(
            dendro_models[feature_columns]
        )

        # Hierarchical clustering
        linked = linkage(
            scaled_data,
            method="ward"
        )

        # Calculate dendrogram structure
        dendro_result = dendrogram(
            linked,
            labels=dendro_models["Car"].tolist(),
            orientation="left",
            no_plot=True
        )

        # Build interactive Plotly dendrogram
        fig8 = go.Figure()

        for xs, ys, color in zip(
            dendro_result["dcoord"],
            dendro_result["icoord"],
            dendro_result["color_list"]
        ):
            fig8.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(width=2),
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        # Add car names to the y-axis
        ordered_labels = dendro_result["ivl"]

        fig8.update_layout(
            title="Hierarchical Car Similarity",
            xaxis_title="Similarity Distance",
            yaxis=dict(
                tickmode="array",
                tickvals=[
                    5 + 10 * i
                    for i in range(len(ordered_labels))
                ],
                ticktext=ordered_labels,
                automargin=True
            ),
            height=max(
                550,
                len(ordered_labels) * 32
            ),
            margin=dict(
                l=190,
                r=30,
                t=70,
                b=60
            )
        )

        st.plotly_chart(
            fig8,
            use_container_width=True
        )

        st.caption(
            "Cars that merge at shorter distances have "
            "more similar price, horsepower and fuel consumption. "
            "Distances are calculated using standardized features."
        )

    else:
        st.warning(
            "Please select more brands or increase the budget "
            "to compare at least two car models."
        )

else:
    st.warning(
        "No cars match your selected brands and budget."
    )



# ==========================================
# GRAPH 7 - ADVANCED SANKEY DIAGRAM
# ==========================================
st.header("Graph 7 - The Death of Manual Transmissions (Flow Analysis)")
st.write(
    "Explore how car production flowed from specific decades into transmission types. "
    "Flows are colored by transmission to highlight the decline of manual gearboxes."
)

base_sankey_df = df.dropna(subset=["Year", "Transmission Type"]).copy()
base_sankey_df = base_sankey_df[
    ~base_sankey_df["Transmission Type"].astype(str).str.lower().isin(["unknown", "nan", ""])
]
base_sankey_df["Decade"] = (base_sankey_df["Year"] // 10 * 10).astype(str) + "s"

sankey_all_brands = sorted(base_sankey_df["Make"].unique())
sankey_all_decades = sorted(base_sankey_df["Decade"].unique())


if "sankey_decades" not in st.session_state:
    st.session_state.sankey_decades = sankey_all_decades


def reset_sankey():
    st.session_state["sankey_brands"] = all_brands.copy()
    st.session_state["sankey_decades"] = sankey_all_decades
    synchronize_brands("sankey_brands")


col1, col2, col3 = st.columns([3, 3, 2])
with col1:
    selected_sankey_brands = st.multiselect(
    "Select Brands:",
    options=all_brands,
    key="sankey_brands",
    on_change=synchronize_brands,
    args=("sankey_brands",)
)
with col2:
    selected_sankey_decades = st.multiselect(
        "Select Decades:",
        options=sankey_all_decades,
        key="sankey_decades"
    )
with col3:
    st.write("")
    st.write("")
    st.button("🔄 Reset Filters", on_click=reset_sankey, key="reset_sankey_btn")

flow_df = base_sankey_df.copy()
if selected_sankey_brands:
    flow_df = flow_df[flow_df["Make"].isin(selected_sankey_brands)]
else:
    flow_df = pd.DataFrame(columns=base_sankey_df.columns)

if selected_sankey_decades and not flow_df.empty:
    flow_df = flow_df[flow_df["Decade"].isin(selected_sankey_decades)]
else:
    flow_df = pd.DataFrame(columns=base_sankey_df.columns)

if not flow_df.empty:
    total_cars = len(flow_df)
    trans_counts = flow_df["Transmission Type"].value_counts()

    kpi_cols = st.columns(len(trans_counts) + 1)
    kpi_cols[0].metric("🚗 Total Vehicles", f"{total_cars:,}")

    for i, (trans, count) in enumerate(trans_counts.items(), start=1):
        pct = (count / total_cars) * 100
        kpi_cols[i].metric(
            label=f"⚙️ {trans}",
            value=f"%{pct:.1f}",
            delta=f"({count:,} cars)",
            delta_color="off"
        )

    st.write("")

    f1 = flow_df.groupby(["Decade", "Transmission Type"]).size().reset_index(name="Count")
    f1.columns = ["Source", "Target", "Count"]
    f1["Trans_Type"] = f1["Target"]

    source_totals = f1.groupby("Source")["Count"].transform("sum")
    f1["Percentage"] = (f1["Count"] / source_totals) * 100

    all_nodes = list(pd.concat([f1["Source"], f1["Target"]]).unique())
    node_indices = {node: i for i, node in enumerate(all_nodes)}

    trans_colors = {
        "MANUAL": {"link": "rgba(214, 39, 40, 0.4)", "node": "#d62728"},
        "AUTOMATIC": {"link": "rgba(44, 160, 44, 0.4)", "node": "#2ca02c"},
        "AUTOMATED_MANUAL": {"link": "rgba(31, 119, 180, 0.4)", "node": "#1f77b4"},
        "DIRECT_DRIVE": {"link": "rgba(255, 127, 14, 0.4)", "node": "#ff7f0e"}
    }

    link_colors = [
        trans_colors.get(trans, {}).get("link", "rgba(169, 169, 169, 0.4)")
        for trans in f1["Trans_Type"]
    ]

    node_colors = []
    for node in all_nodes:
        if node.endswith("s"):
            node_colors.append("#34495e")
        elif node in trans_colors:
            node_colors.append(trans_colors[node]["node"])
        else:
            node_colors.append("#8e44ad")

    fig7 = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=node_colors
        ),
        link=dict(
            source=f1["Source"].map(node_indices),
            target=f1["Target"].map(node_indices),
            value=f1["Count"],
            color=link_colors,
            customdata=np.stack((f1["Percentage"], f1["Trans_Type"]), axis=-1),
            hovertemplate=(
                "<b>%{source.label} → %{target.label}</b><br>"
                "Vehicle Count: <b>%{value:,}</b><br>"
                "Share from %{source.label}: <b>%{customdata[0]:.1f}%</b><br>"
                "<extra></extra>"
            )
        )
    )])

    fig7.update_layout(
        title="Transmission Type Distribution by Decade",
        height=600,
        font=dict(size=13),
        margin=dict(t=50, b=25, l=25, r=25)
    )

    st.plotly_chart(fig7, use_container_width=True)

    st.info(
        "💡 *Insight:* Notice how the thick green flow of Automatic transmissions increasingly "
        "dominates the later decades, while the red flow of Manual transmissions shrinks considerably."
    )
else:
    st.warning("No flow data available.")

st.divider()


# ==========================================
# GRAPH 8 - 100% STACKED AREA CHART
# The Evolution & Diversification of Fuel Technologies
# ==========================================
st.header("Graph 8 - The Evolution of Fuel Technologies (Full Spectrum)")
st.write(
    "Explore how the automotive market evolved from a simple Regular vs. Premium split "
    "into a diverse ecosystem of Recommended Premium, Flex-Fuel (E85), Diesel, and Electric powertrains."
)

def categorize_full_fuel(fuel: str) -> str:
    f = str(fuel).lower().strip()
    if "flex-fuel" in f:
        return "Flex-Fuel (E85)"
    elif "premium" in f and "required" in f:
        return "Premium (Required)"
    elif "premium" in f and "recommended" in f:
        return "Premium (Recommended)"
    elif "premium" in f:
        return "Premium (Other)"
    elif "regular" in f:
        return "Regular Unleaded"
    elif "diesel" in f:
        return "Diesel"
    elif "electric" in f:
        return "Electric"
    elif "natural gas" in f:
        return "Natural Gas"
    return "Other"

base_fuel_df = df.dropna(subset=["Year", "Engine Fuel Type"]).copy()
base_fuel_df["Detailed Fuel"] = base_fuel_df["Engine Fuel Type"].apply(categorize_full_fuel)

fuel_all_brands = sorted(base_fuel_df["Make"].unique())
fuel_all_types = sorted(base_fuel_df["Detailed Fuel"].unique())
min_year = int(base_fuel_df["Year"].min())
max_year = int(base_fuel_df["Year"].max())


if "g8_fuels" not in st.session_state:
    st.session_state.g8_fuels = fuel_all_types
if "g8_years" not in st.session_state:
    st.session_state.g8_years = (min_year, max_year)

def reset_g8_filters():
    st.session_state["g8_brands"] = all_brands.copy()
    st.session_state["g8_fuels"] = fuel_all_types
    st.session_state["g8_years"] = (min_year, max_year)
    synchronize_brands("g8_brands")

col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
with col1:
    selected_g8_brands = st.multiselect(
    "Select Brands:",
    options=all_brands,
    key="g8_brands",
    on_change=synchronize_brands,
    args=("g8_brands",)
)
with col2:
    selected_g8_fuels = st.multiselect(
        "Select Fuel Types:",
        options=fuel_all_types,
        key="g8_fuels"
    )
with col3:
    selected_g8_years = st.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        key="g8_years"
    )
with col4:
    st.write("")
    st.write("")
    st.button("🔄 Reset Filters", on_click=reset_g8_filters, key="reset_g8_btn")


plot_fuel_df = base_fuel_df[
    (base_fuel_df["Make"].isin(selected_g8_brands)) &
    (base_fuel_df["Detailed Fuel"].isin(selected_g8_fuels)) &
    (base_fuel_df["Year"] >= selected_g8_years[0]) &
    (base_fuel_df["Year"] <= selected_g8_years[1])
].copy()

if not plot_fuel_df.empty:
    total_fuel_cars = len(plot_fuel_df)
    fuel_counts = plot_fuel_df["Detailed Fuel"].value_counts()

    kpi_cols = st.columns(min(len(fuel_counts) + 1, 7))
    kpi_cols[0].metric("🚗 Total Vehicles", f"{total_fuel_cars:,}")

    for i, (f_type, count) in enumerate(fuel_counts.head(6).items(), start=1):
        pct = (count / total_fuel_cars) * 100
        kpi_cols[i].metric(
            label=f"⛽ {f_type}",
            value=f"%{pct:.1f}",
            delta=f"({count:,} cars)",
            delta_color="off"
        )

    st.write("")

    yearly_fuel = (
        plot_fuel_df.groupby(["Year", "Detailed Fuel"])
        .size()
        .reset_index(name="Count")
    )
    yearly_fuel["Year_Total"] = yearly_fuel.groupby("Year")["Count"].transform("sum")
    yearly_fuel["Market_Share"] = (yearly_fuel["Count"] / yearly_fuel["Year_Total"]) * 100

    fuel_color_map = {
        "Regular Unleaded": "#27ae60",
        "Premium (Recommended)": "#f39c12",
        "Premium (Required)": "#c0392b",
        "Flex-Fuel (E85)": "#8e44ad",
        "Diesel": "#34495e",
        "Electric": "#00bcd4",
        "Natural Gas": "#16a085"
    }

    fig8 = px.area(
        yearly_fuel,
        x="Year",
        y="Market_Share",
        color="Detailed Fuel",
        custom_data=["Count"],
        title="Market Share of Detailed Fuel Technologies Over Time (100% Stacked)",
        labels={
            "Market_Share": "Market Share (%)",
            "Year": "Production Year",
            "Detailed Fuel": "Fuel Technology"
        },
        color_discrete_map=fuel_color_map
    )

    fig8.update_traces(
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Market Share: <b>%{y:.1f}%</b><br>"
            "Vehicle Count: <b>%{customdata[0]:,}</b><extra></extra>"
        )
    )

    fig8.update_layout(
        height=600,
        yaxis=dict(range=[0, 100]),
        hovermode="x unified",
        margin=dict(t=50, b=25, l=25, r=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig8, use_container_width=True)

    st.info(
        "💡 *Insight:* Splitting fuel data into its true subcategories reveals two major trends: "
        "First, after 2000, automakers heavily expanded *Premium (Recommended)* (orange) alongside *Premium (Required)* (red). "
        "Second, alternative fuels gained significant traction in the 2000s and 2010s."
    )
else:
    st.warning("No valid fuel data available. Please adjust your filters.")

st.divider()


# ==========================================
# GRAPH 9 - ML PROBABILITY CURVE (LOGISTIC REGRESSION)
# The Premium Fuel Threshold
# ==========================================
st.header("Graph 9 - The Premium Fuel Threshold (AI Probability Curve)")
st.write(
    "At what exact horsepower does an engine cross the line from standard to luxury? "
    "We trained a Logistic Regression Machine Learning model to calculate the exact probability "
    "of a car requiring expensive Premium fuel based on its power output."
)

ml_df = df[df["Fuel Category"].isin(["Regular Fuel", "Premium Fuel"])].copy()
ml_df = ml_df.dropna(subset=["Engine HP", "Engine Cylinders"])
ml_df["Is_Premium"] = np.where(ml_df["Fuel Category"] == "Premium Fuel", 1, 0)

# HATAYI ÖNLEMEK İÇİN: Sadece 3 ve 10 olmayan silindirleri alıyoruz
cyl_options = sorted(ml_df["Engine Cylinders"].astype(int).unique())
cyl_options = [cyl for cyl in cyl_options if cyl not in [3, 10]]

default_cyl_index = cyl_options.index(4) if 4 in cyl_options else 0

selected_cylinders = st.selectbox(
    "Select Engine Cylinder Count to inspect the AI's exact horsepower threshold:",
    options=cyl_options,
    index=default_cyl_index,
    key="ml_fuel_cylinders"
)

plot_ml_df = ml_df[ml_df["Engine Cylinders"] == selected_cylinders].copy()

# HATA ÖNLEYİCİ GÜVENLİK DUVARI:
# Hem veri sayısı 20'den büyük olmalı, Hem de veri seti içerisinde mutlaka iki sınıf (Regular ve Premium) olmalı
if len(plot_ml_df) >= 20 and plot_ml_df["Is_Premium"].nunique() > 1:
    
    X = plot_ml_df[["Engine HP"]]
    y = plot_ml_df["Is_Premium"]
    
    clf_log = LogisticRegression(random_state=42)
    clf_log.fit(X, y)
    
    min_hp = plot_ml_df["Engine HP"].min()
    max_hp = plot_ml_df["Engine HP"].max()
    x_smooth = np.linspace(min_hp, max_hp, 300).reshape(-1, 1)
    
    y_prob = clf_log.predict_proba(x_smooth)[:, 1]
    
    threshold_hp = None
    if clf_log.coef_[0][0] != 0:
        threshold_hp = -clf_log.intercept_[0] / clf_log.coef_[0][0]
        
    fig9 = go.Figure()
    
    fig9.add_trace(
        go.Scatter(
            x=x_smooth.flatten(),
            y=y_prob * 100,
            mode="lines",
            name="AI Probability Curve",
            line=dict(color="black", width=5),
            hovertemplate="Horsepower: %{x:.0f} HP<br>Premium Probability: <b>%{y:.1f}%</b><extra></extra>"
        )
    )
    
    if threshold_hp and (min_hp < threshold_hp < max_hp):
        fig9.add_vrect(
            x0=min_hp, x1=threshold_hp,
            fillcolor="#28a745", opacity=0.15,
            layer="below", line_width=0
        )
        fig9.add_vrect(
            x0=threshold_hp, x1=max_hp,
            fillcolor="#dc3545", opacity=0.15,
            layer="below", line_width=0
        )
        fig9.add_vline(
            x=threshold_hp, line_width=3, line_dash="dash", line_color="black",
            annotation_text=f"AI Threshold: {threshold_hp:.0f} HP", 
            annotation_position="top left"
        )
        
        st.metric(
            label="🎯 AI Discovered Threshold (50% Probability)",
            value=f"{threshold_hp:.0f} HP",
            delta="After this point, Premium Fuel is practically required",
            delta_color="inverse"
        )
    
    jitter_y = plot_ml_df["Is_Premium"] * 100 + np.random.uniform(-3, 3, size=len(plot_ml_df))
    colors = np.where(plot_ml_df["Is_Premium"] == 1, "#dc3545", "#28a745")
    labels = np.where(plot_ml_df["Is_Premium"] == 1, "Premium Fuel", "Regular Fuel")
    
    fig9.add_trace(
        go.Scatter(
            x=plot_ml_df["Engine HP"],
            y=jitter_y,
            mode="markers",
            name="Real Cars",
            marker=dict(color=colors, size=7, opacity=0.6, line=dict(width=1, color="white")),
            text=plot_ml_df["Make"] + " " + plot_ml_df["Model"] + " (" + labels + ")",
            hovertemplate="<b>%{text}</b><br>Horsepower: %{x} HP<extra></extra>",
            showlegend=False
        )
    )

    fig9.update_layout(
        title=f"Probability of Requiring Premium Fuel ({selected_cylinders}-Cylinder Engines)",
        xaxis_title="Engine Horsepower (HP)",
        yaxis_title="Probability (%)",
        yaxis=dict(range=[-5, 105], tickvals=[0, 25, 50, 75, 100], ticktext=["0%", "25%", "50%", "75%", "100%"]),
        height=600,
        margin=dict(t=50, b=25, l=25, r=25)
    )
    
    st.plotly_chart(fig9, use_container_width=True)

    st.info(
        "💡 *Final Insight:* The thick black S-curve represents the Artificial Intelligence calculating the odds. "
        f"Notice how accurately the AI identifies the tipping point. The moment an automaker pushes a {selected_cylinders}-cylinder "
        "engine past the dashed threshold line, the physical limits of the engine force it to demand Premium fuel. "
    )
else:
    st.warning(f"Not enough variance (requires both Regular and Premium fuel examples) to train a reliable AI model for {selected_cylinders}-cylinder engines.")

st.divider()






















#python -m streamlit run app.py