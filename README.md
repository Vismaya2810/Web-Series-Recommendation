# Web-Series-Recommendation

A modern, Netflix-style web application for personalized Malayalam web series recommendations powered by machine learning algorithms.
Check out the app: https://malayalam-web-series-recommendation.streamlit.app/

## Features

- 🎬 **Hybrid Recommendations**: Combines content-based and collaborative filtering for optimal suggestions
- 📝 **Content-Based Filtering**: Find series similar to your favorites based on genre, content, and cast
- 👥 **Collaborative Filtering**: Discover series loved by users with similar tastes
- 📊 **Model Comparison**: Evaluate and compare different recommendation models with detailed metrics
- 🎨 **Netflix-like UI**: Beautiful, dark-themed interface for an immersive experience

## Technologies Used

- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning algorithms (TF-IDF, SVD, Cosine Similarity)
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Web-Series-Recommendation.git
cd Web-Series-Recommendation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

### Step 1: Push to GitHub

1. Create a new repository on GitHub (if you haven't already)

2. Initialize git in your project directory (if not already done):
```bash
git init
git add .
git commit -m "Initial commit"
```

3. Add your GitHub repository as remote:
```bash
git remote add origin https://github.com/yourusername/Web-Series-Recommendation.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click on "New app"
4. Select your repository: `yourusername/Web-Series-Recommendation`
5. Set the main file path: `app.py`
6. Set the branch: `main` (or your default branch)
7. Click "Deploy"

Your app will be live at: `https://yourusername-web-series-recommendation.streamlit.app`

## Project Structure

```
Web-Series-Recommendation/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
├── .streamlit/           # Streamlit configuration
│   └── config.toml       # Streamlit theme and settings
└── data/
    └── dataset.csv       # Web series dataset
```

## Usage

1. **Home Page**: Welcome screen with project overview
2. **Recommendations Hub**: Choose your preferred recommendation type
3. **Hybrid Recommendations**: Enter a User ID (1-100) to get personalized hybrid recommendations
4. **Content-Based Recommendations**: Select a series to find similar ones
5. **Collaborative Filtering**: Enter a User ID to get recommendations based on similar users
6. **Model Comparison**: View performance metrics comparing all models

## Algorithm Details

- **Content-Based Filtering (CBF)**: Uses TF-IDF vectorization and cosine similarity to find series with similar content features
- **Collaborative Filtering (CF)**: Uses Singular Value Decomposition (SVD) to predict user ratings
- **Hybrid Model**: Combines CBF and CF with a weighted average (alpha parameter)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions, issues, and feature requests are welcome!

## Author

Created with ❤️ for Malayalam web series enthusiasts
