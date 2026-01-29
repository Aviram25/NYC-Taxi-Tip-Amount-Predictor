# 🚕 NYC Taxi Tip Predictor - Streamlit Dashboard

A beautiful, interactive web application for predicting and visualizing NYC taxi tip amounts using machine learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

### 📈 Model Performance Dashboard
- Compare multiple ML models (Random Forest, XGBoost, Tuned RF)
- View accuracy and F1 scores with interactive charts
- Detailed performance metrics and comparisons

### 🎯 Interactive Tip Predictor
- Input trip details through an intuitive form
- Get real-time predictions with confidence scores
- View probability distributions across all tip classes
- See expected tip amounts and percentages

### 📊 Data Insights
- Visualize tip class distribution with bar and pie charts
- Analyze feature distributions by tip class
- Interactive violin plots for feature exploration
- Summary statistics and key metrics

### 🔍 Feature Analysis
- Feature importance visualization
- Correlation heatmap between features
- Top features ranked by importance
- Detailed insights on feature relationships

### 📉 Confusion Matrix
- Interactive confusion matrix visualization
- Per-class performance metrics
- Precision, recall, and F1-score breakdowns
- Model comparison capabilities

### 💡 Business Insights
- Actionable recommendations for drivers
- App development suggestions
- Analytics and monitoring guidelines
- Key performance indicators with gauges

## 🚀 Quick Start

### Installation

1. **Clone or download the files**
   ```bash
   # Make sure you have these files:
   # - streamlit_app.py
   # - requirements_streamlit.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

### Running the App

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements_streamlit.txt`:
  - streamlit >= 1.28.0
  - plotly >= 5.17.0
  - pandas >= 2.0.0
  - numpy >= 1.24.0

## 🎯 Tip Class Definitions

The model predicts 5 different tip ranges:

| Class | Range | Description |
|-------|-------|-------------|
| 🔴 No Tip | $0 | No tip recorded |
| 🟡 Low | $0-$3 | Small tip amount |
| 🟢 Medium | $3-$6 | Average tip amount |
| 🔵 High | $6-$10 | Above average tip |
| 🟣 Very High | >$10 | Exceptional tip |

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 71.34% | 69.20% |
| **XGBoost** | **74.37%** | **73.25%** |
| Tuned RF | 73.65% | 72.26% |

## 🎨 Features Used

The model uses 8 engineered features:

1. **fare_amount** - Total fare charged (35% importance)
2. **trip_distance** - Distance traveled (22% importance)
3. **trip_duration** - Trip length in minutes (18% importance)
4. **payment_type** - Payment method (12% importance)
5. **pickup_hour** - Hour of pickup (6% importance)
6. **pickup_day** - Day of week (4% importance)
7. **pickup_month** - Month of year (2% importance)
8. **is_weekend** - Weekend indicator (1% importance)

## 🖼️ Screenshots

### Main Dashboard
The app features a modern, gradient-styled interface with:
- Interactive tab navigation
- Real-time predictions
- Beautiful data visualizations
- Responsive design

### Prediction Interface
- Intuitive sliders and selectors
- Instant feedback
- Confidence scores
- Probability distributions

## 🔧 Customization

### Change Theme Colors
Edit the CSS in the `st.markdown()` section at the top of `streamlit_app.py`:

```python
# Change gradient colors
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Modify Prediction Logic
Update the `predict_tip_class()` function to:
- Load your own trained model
- Adjust prediction logic
- Change confidence calculations

### Add New Visualizations
Add new charts in any tab section using Plotly:

```python
fig = go.Figure()
# Add your visualization
st.plotly_chart(fig, use_container_width=True)
```

## 📱 Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `streamlit_app.py` as the main file
   - Click "Deploy"

### Deploy to Heroku

1. **Create Procfile**
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Deploy to AWS/GCP/Azure

Use Docker with this `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🧪 Testing

Test the app locally:

```bash
# Run the app
streamlit run streamlit_app.py

# Test different scenarios:
# 1. Enter high fare amount → Expect high tip prediction
# 2. Select "Cash" payment → Expect "No Tip" prediction
# 3. Try weekend vs weekday → See difference in confidence
```

## 📈 Performance Optimization

### For Large Datasets
If loading actual data, use caching:

```python
@st.cache_data
def load_data():
    return pd.read_parquet('data.parquet')
```

### For Model Loading
Cache model loading:

```python
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))
```

## 🐛 Troubleshooting

### Common Issues

**Port already in use:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Module not found:**
```bash
pip install --upgrade -r requirements_streamlit.txt
```

**Plotly not displaying:**
- Clear browser cache
- Try a different browser
- Check JavaScript is enabled

### Performance Issues

- Reduce sample sizes in violin plots
- Use `st.cache_data` for data loading
- Limit number of data points in visualizations

## 📚 Documentation

### Streamlit
- [Official Docs](https://docs.streamlit.io)
- [Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)
- [Gallery](https://streamlit.io/gallery)

### Plotly
- [Python Docs](https://plotly.com/python/)
- [Graph Reference](https://plotly.com/python/reference/)
- [Examples](https://plotly.com/python/basic-charts/)

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- Streamlit for the amazing framework
- Plotly for beautiful visualizations

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check [Streamlit Community](https://discuss.streamlit.io)
- Review the documentation

## 🔮 Future Enhancements

- [ ] Real-time model training
- [ ] Advanced filtering options
- [ ] Export predictions to CSV
- [ ] A/B testing framework
- [ ] Multi-language support
- [ ] Dark mode toggle
- [ ] Mobile app version
- [ ] API integration

## 📊 Dataset Information

- **Source:** NYC Taxi & Limousine Commission
- **Records:** 797,904 taxi trips
- **Time Period:** [Specify based on your data]
- **Features:** 8 engineered features
- **Target:** 5-class tip range classification

---

**Built with ❤️ using Streamlit and Plotly**

For more information, visit the [project repository](#) or [documentation](#).
