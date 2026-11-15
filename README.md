# Car Data Visualization Dashboard

An interactive exploratory data visualization dashboard built with **Python, Streamlit, and Plotly** for the CEN445 Introduction to Data Visualization course. The project analyzes vehicle prices, performance, fuel efficiency, transmission types, and fuel technologies using nine visualizations and two machine learning approaches.

## Dataset

- **Name:** Car Features and MSRP
- **Source:** [CooperUnion Car Dataset on Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)
- **Local file:** `car_data.csv` (semicolon-separated in this project)
- **Main attributes:** Make, Model, Year, MSRP, Engine HP, Engine Cylinders, Engine Fuel Type, Transmission Type, Driven_Wheels, city mpg, and highway MPG.

## Data preprocessing

The dashboard removes records with missing values in selected essential columns, identifies potential outliers using the interquartile range (IQR) method, and retains potentially valid extreme vehicle values. It also groups fuel types into categories and converts MPG into L/100 km for consumption comparisons.

## Visualizations

| Visualization | Purpose |
| --- | --- |
| Heatmap | Compare average city and highway fuel consumption by brand. |
| Parallel Coordinates | Explore price, horsepower, and fuel-consumption trade-offs. |
| 3D K-Means Clustering | Group similar vehicles by price, horsepower, and consumption. |
| Bubble Chart | Compare car models by price, power, and fuel use. |
| Sunburst | Explore brand, model, transmission, and drivetrain hierarchies. |
| Hierarchical Clustering Dendrogram | Examine similarity among car models. |
| Sankey Diagram | Show transmission-type distributions across decades. |
| Stacked Area Chart | Compare fuel-technology shares across production years. |
| Logistic Regression Probability Curve | Model premium-fuel classification probability using horsepower for selected cylinder counts. |

Charts 1–8 share synchronized brand selections. Other interactive features include price and year sliders, model selection, tooltips, zooming, and chart-specific filters.

## Installation and usage

1. Install Python and download or clone this repository.
2. Keep `app.py` and `car_data.csv` in the same directory.
3. In a terminal opened in the project directory, run:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The app will open in a browser (typically at `http://localhost:8501`).

## Required files

- `app.py` — complete Streamlit dashboard
- `car_data.csv` — project dataset
- `requirements.txt` — Python dependencies
- `README.md` — documentation
- `Car_Data_Visualization_Report_Final.pdf` — one-page English report (optional repository attachment)

## Team and individual contributions

**Ahmet Eren Bozduman — 2022556014**  
Developed the Heatmap, Parallel Coordinates, and 3D K-Means Clustering visualizations; implemented analytical filters, refined chart presentation, and addressed technical and visualization issues in these components.

**Sıla Toker — 2022556059**  
Developed the Bubble Chart, Sunburst Chart, and Hierarchical Clustering Dendrogram; designed vehicle-model comparisons, implemented interactive controls, and resolved filtering and visualization issues in these components.

**Muhammed Yaşar Karaçam — 2022556037**  
Developed the Sankey Diagram, Stacked Area Chart, and Logistic Regression Probability Curve; implemented fuel and transmission analyses, chart-specific controls, and troubleshooting for these components.

## Notes

This dashboard uses a local CSV dataset and does **not** require an API key. Do not upload passwords, access tokens, `.env` files, or private credentials. Dataset attribution is provided above; consult the original source for applicable reuse terms.
