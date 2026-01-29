# 🚕 NYC Taxi Tip Predictor - Interactive Streamlit Dashboard

An interactive web application for visualizing and predicting NYC taxi tip amounts using machine learning models.

## 📋 Features

### 1. **Model Performance Dashboard**
- Compare Random Forest, XGBoost, and Tuned Random Forest models
- View accuracy and F1 scores with visual comparisons
- Interactive confusion matrix visualization

### 2. **Interactive Tip Predictor**
- Input trip details (fare, distance, duration, time, payment type)
- Get real-time predictions with confidence scores
- View probability distribution across all tip classes

### 3. **Data Analysis**
- Visualize tip class distribution
- Analyze feature distributions by tip class
- Interactive violin plots for feature exploration

### 4. **Feature Importance**
- View which features most influence predictions
- Correlation heatmap between features
- Top features with importance scores

### 5. **Insights & Recommendations**
- Model performance analysis
- Business recommendations
- Key findings with interactive gauges

## 🎯 Tip Classes

The model predicts 5 tip ranges:
- 🔴 **No Tip**: $0
- 🟡 **Low**: $0 - $3
- 🟢 **Medium**: $3 - $6
- 🔵 **High**: $6 - $10
- 🟣 **Very High**: >$10

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 71.34% | 69.20% |
| XGBoost | **74.37%** | **73.25%** |
| Tuned RF | 73.65% | 72.26% |

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install streamlit plotly pandas numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run the Application

```bash
streamlit run tip_predictor_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Navigation

1. **📈 Model Performance**: Compare different models and view confusion matrices
2. **🎯 Interactive Predictor**: Make predictions with custom trip parameters
3. **📊 Data Analysis**: Explore the dataset and class distributions
4. **🔍 Feature Importance**: Understand which features drive predictions
5. **💡 Insights**: View key findings and recommendations

## 📈 Features Used in the Model

1. **fare_amount** - The fare charged for the trip
2. **trip_distance** - Distance traveled in miles
3. **trip_duration** - Duration of the trip in minutes
4. **payment_type** - Method of payment (Credit Card, Cash, etc.)
5. **pickup_hour** - Hour of the day when trip started
6. **pickup_day** - Day of the week
7. **pickup_month** - Month of the year
8. **is_weekend** - Whether the trip occurred on a weekend

## 🎨 Visualizations

The dashboard includes:
- **Bar charts** for model comparison
- **Confusion matrices** for classification performance
- **Pie charts** for class distribution
- **Violin plots** for feature distributions
- **Heatmaps** for feature correlations
- **Gauge charts** for key metrics
- **Interactive prediction interface**

## 📝 Technical Details

### Models Implemented
1. **Random Forest Classifier** (20 trees, max depth 5)
2. **XGBoost Classifier** (default parameters)
3. **Tuned Random Forest** (50-100 trees, max depth 5-10)

### Dataset
- **Source**: NYC Taxi Dataset
- **Total Records**: 797,904 trips
- **Train/Test Split**: 80/20

### Framework
- **Backend**: PySpark for distributed processing
- **Visualization**: Streamlit + Plotly
- **Models**: scikit-learn, XGBoost

## 🎯 Key Insights

1. **XGBoost performs best** with 74.4% accuracy
2. **Fare amount is the strongest predictor** (35% importance)
3. **Medium tips ($3-$6) are most common** (41.2% of rides)
4. **Weekend rides** tend to have slightly higher tips
5. **Payment type matters** - cash tips often not recorded

## 🔧 Customization

You can customize the app by:
- Adjusting color schemes in the CSS section
- Modifying prediction logic in the Interactive Predictor tab
- Adding new visualizations or metrics
- Changing the model parameters

## 📦 File Structure

```
├── tip_predictor_app.py       # Main Streamlit application
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── Tip_Amount_Model_2_2.ipynb # Original model notebook
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run tip_predictor_app.py --server.port 8502
```

### Module Not Found
```bash
pip install --upgrade streamlit plotly pandas numpy
```

### Slow Performance
- Reduce sample sizes in the Data Analysis tab
- Close other browser tabs
- Refresh the page

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest new features
- Improve visualizations
- Optimize performance

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- NYC Taxi & Limousine Commission for the dataset
- PySpark for distributed computing capabilities
- Streamlit for the amazing visualization framework

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Streamlit documentation: https://docs.streamlit.io
3. Review Plotly documentation: https://plotly.com/python/

---

**Built with ❤️ using Streamlit, Plotly, and Python**
