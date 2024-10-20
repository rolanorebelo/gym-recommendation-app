# Gym Recommendation App

The **Gym Recommendation App** helps users find gyms tailored to their preferences, using location data and personalized reviews. By leveraging the Google Places API and NLP techniques, the app provides a list of nearby gyms, rates them based on user-defined preferences, and displays them on an interactive map.

## Features

- **Location-Based Search**: Find gyms within a specified radius of a given location using the Google Places API.
- **Preference-Based Recommendations**: Users can rate the importance of various factors like hygiene, equipment quality, and crowd level to get customized gym ratings.
- **Custom Filters**: Add personalized filters (e.g., "trainer quality", "affordability") to refine gym searches.
- **Gym Reviews Analysis**: Reviews are analyzed using NLP to calculate ratings tailored to the user’s preferences.
- **Interactive Map**: View gym locations on an interactive map using Folium.
- **Data Visualization**: Display top 10 gyms based on the tailored rating using Plotly.
- **Downloadable Results**: Download gym recommendations as a CSV file.

## Tech Stack

- **Streamlit**: Used for building the web app UI.
- **Google Maps API**: Fetch nearby gym data and locations.
- **Folium**: Interactive maps for gym locations.
- **TextBlob**: Used for natural language processing (NLP) and sentiment analysis of gym reviews.
- **Plotly**: Data visualization for gym ratings.
- **Pandas**: Data handling and manipulation.

## Prerequisites

- **Google Maps API Key**: This app requires a valid API key to access Google Maps services. Obtain an API key from the [Google Cloud Console](https://console.cloud.google.com/).
- Python 3.x

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/gym-recommendation-app.git
   cd gym-recommendation-app
   ```

2. **Create a virtual environment:**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. ***Set up the .env file: Create a .env file in the root directory with the following content:***
   ```bash
   API_KEY=your_google_maps_api_key
   ```

5. ***Run the app***
```bash
streamlit run app.py
```

7. ***Access the app:***
   The app will run locally and can be accessed via the browser at http://localhost:8501.

   ## Usage

1. Enter your desired location (e.g., "New York, NY").
2. Rate the importance of factors such as hygiene, equipment quality, and atmosphere.
3. Add custom filters for even more personalized results.
4. View tailored gym recommendations based on your preferences, explore them on a map, and download the recommendations as a CSV file.

## Screenshots

![App Interface](path/to/screenshot1.png)
*Main interface of the Gym Recommendation App*

![Map View](path/to/screenshot2.png)
*Interactive map showing gym locations*

![Results](path/to/screenshot3.png)
*Tailored gym recommendations and ratings*

## Future Improvements

- Integrate more review analysis techniques for better accuracy.
- Add more filters for better gym recommendations (e.g., gym opening hours, gym classes).
- Implement user authentication to save user preferences and searches.
- Enhance the user interface for a more intuitive experience.
- Incorporate real-time data updates for gym information and reviews.
