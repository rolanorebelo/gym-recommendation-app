import streamlit as st
import requests
import pandas as pd
import googlemaps
import nltk
import folium
from streamlit_folium import folium_static
import plotly.express as px
from textblob import TextBlob
import corpora
from dotenv import load_dotenv
import os

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')

# Load environment variables from .env file
load_dotenv()

nltk.download('punkt', quiet=True)

# Google Maps API Key (make sure to replace with your own key)
API_KEY = os.getenv('API_KEY')
gmaps = googlemaps.Client(key=API_KEY)

preference_keywords = {
    'hygiene': ['clean', 'sanitary', 'hygienic'],
    'equipment_quality': ['equipment', 'machines', 'weights', 'dumbbells'],
    'less_crowd': ['crowded', 'busy', 'people'],
    'trainer_knowledge': ['trainer', 'coach', 'instructor', 'knowledgeable'],
    'price': ['price', 'cost', 'expensive', 'affordable'],
    'amenities': ['pool', 'sauna', 'classes', 'locker'],
    'atmosphere': ['atmosphere', 'environment', 'vibe', 'friendly']
}

@st.cache_data(ttl=3600)
def get_nearby_gyms(lat, lng, radius, place_type):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius={radius}&type={place_type}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        st.error("Error fetching gyms: " + response.json().get('error_message'))
        return []

@st.cache_data(ttl=3600)
def get_reviews(place_id):
    details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,rating,reviews,formatted_address,formatted_phone_number,website&key={API_KEY}"
    details_response = requests.get(details_url)
    if details_response.status_code == 200:
        return details_response.json().get('result', {})
    else:
        st.error("Error fetching reviews: " + details_response.json().get('error_message'))
        return {}

@st.cache_data(ttl=3600)
def get_lat_lng(location_name):
    geocode_result = gmaps.geocode(location_name)
    if geocode_result:
        lat_lng = geocode_result[0]['geometry']['location']
        return lat_lng['lat'], lat_lng['lng']
    else:
        st.error("Location not found.")
        return None, None

def calculate_preference_rating(review_text, preferences, custom_filters):
    blob = TextBlob(review_text.lower())
    score = 0
    for pref, importance in preferences.items():
        for keyword in preference_keywords[pref]:
            if keyword in blob.words:
                score += importance  # Weight by user preference
    
    # Include custom filters in the calculation
    for keyword, importance in custom_filters.items():
        if keyword.lower() in blob.words:
            score += importance

    return score

def get_gym_reviews_and_ratings(lat, lng, radius, preferences, custom_filters):
    gyms = get_nearby_gyms(lat, lng, radius, 'gym')
    all_gym_ratings = []
    
    progress_bar = st.progress(0)
    for i, gym in enumerate(gyms):
        place_id = gym['place_id']
        gym_details = get_reviews(place_id)
        gym_name = gym_details.get('name', '')
        gym_address = gym_details.get('formatted_address', '')
        gym_phone = gym_details.get('formatted_phone_number', '')
        gym_website = gym_details.get('website', '')
        reviews = gym_details.get('reviews', [])

        total_score = 0
        for review in reviews:
            review_text = review.get('text', '')
            review_score = calculate_preference_rating(review_text, preferences, custom_filters)
            total_score += review_score

        avg_rating = total_score / len(reviews) if reviews else 0
        all_gym_ratings.append({
            'Gym Name': gym_name,
            'Tailored Rating': avg_rating,
            'Address': gym_address,
            'Phone': gym_phone,
            'Website': gym_website,
            'Latitude': gym['geometry']['location']['lat'],
            'Longitude': gym['geometry']['location']['lng']
        })
        
        progress_bar.progress((i + 1) / len(gyms))

    return pd.DataFrame(all_gym_ratings)

def create_map(gyms, lat, lng):
    m = folium.Map(location=[lat, lng], zoom_start=13)
    for _, gym in gyms.iterrows():
        folium.Marker(
            [gym['Latitude'], gym['Longitude']],
            popup=f"{gym['Gym Name']}<br>Rating: {gym['Tailored Rating']:.2f}<br><a href='{gym['Website']}' target='_blank'>Website</a>"
        ).add_to(m)
    return m

# Apply custom CSS styling for title and subtitle
st.markdown(
    """
    <style>
    /* Title styling */
    h1 {
        color: #4682B4 !important;  /* Nice shade of blue */
        text-align: center;         /* Center alignment */
        font-size: 3em;             /* Larger font size */
    }
    
    /* Subtitle styling */
    h2 {
        color: gray !important;
        text-align: center;         /* Center alignment */
        font-size: 1.5em;           /* Slightly smaller font size */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle with styling
st.markdown(
    """
    <h1>FitFindr</h1>
    <h2>Enhanced Gym Recommendation App</h2>
    """,
    unsafe_allow_html=True,
)

# User input for location
location = st.text_input("Enter the location (e.g., 'New York, NY')")

# User input for preferences
st.subheader("Rate the importance of the following factors (1 = Low, 5 = High):")
hygiene = st.slider("Hygiene", 1, 5, 3)
equipment_quality = st.slider("Equipment Quality", 1, 5, 3)
less_crowd = st.slider("Less Crowd", 1, 5, 3)
trainer_knowledge = st.slider("Trainer Knowledge", 1, 5, 3)
price = st.slider("Price", 1, 5, 3)
amenities = st.slider("Amenities", 1, 5, 3)
atmosphere = st.slider("Atmosphere", 1, 5, 3)

# Custom Filters
st.subheader("Custom Filters")
custom_filters = {}

# Allow users to add custom filters
col1, col2, col3 = st.columns(3)
with col1:
    custom_keyword = st.text_input("Enter a custom keyword")
with col2:
    custom_importance = st.slider("Importance", 1, 5, 3, key="custom_importance")
with col3:
    if st.button("Add Custom Filter"):
        if custom_keyword:
            custom_filters[custom_keyword] = custom_importance
            st.success(f"Added custom filter: {custom_keyword}")
        else:
            st.error("Please enter a keyword")

# Display current custom filters
if custom_filters:
    st.write("Current Custom Filters:")
    for keyword, importance in custom_filters.items():
        st.write(f"- {keyword}: Importance {importance}")

# Additional filters
st.subheader("Additional Filters")
min_rating = st.slider("Minimum Tailored Rating", 0.0, 5.0, 0.0, 0.1)
max_distance = st.slider("Maximum Distance (km)", 1, 10, 5)

# Button to get recommendations
if st.button("Get Gym Recommendations"):
    if not location.strip():
        st.error("Please enter a location before getting recommendations.")
    else:
        lat, lng = get_lat_lng(location)

        if lat is not None and lng is not None:
            radius = max_distance * 1000  # Convert km to meters
            preferences = {
                'hygiene': hygiene,
                'equipment_quality': equipment_quality,
                'less_crowd': less_crowd,
                'trainer_knowledge': trainer_knowledge,
                'price': price,
                'amenities': amenities,
                'atmosphere': atmosphere
            }

            with st.spinner("Fetching gym recommendations..."):
                try:
                    # Get gym reviews and calculate tailored ratings
                    gym_ratings_df = get_gym_reviews_and_ratings(lat, lng, radius, preferences, custom_filters)

                    # Apply filters
                    filtered_gyms = gym_ratings_df[gym_ratings_df['Tailored Rating'] >= min_rating]

                    # Sort gyms by tailored rating
                    sorted_gyms = filtered_gyms.sort_values(by='Tailored Rating', ascending=False).reset_index(drop=True)

                    if sorted_gyms.empty:
                        st.warning("No gyms found matching your criteria. Try adjusting your filters or increasing the search radius.")
                    else:
                        # Display all gyms
                        st.subheader("Gyms tailored to your preferences:")
                        st.dataframe(sorted_gyms)

                        # Map visualization
                        st.subheader("Gym Locations")
                        map = create_map(sorted_gyms, lat, lng)
                        folium_static(map)

                        # Data visualization
                        st.subheader("Top 10 Gyms by Tailored Rating")
                        fig = px.bar(sorted_gyms.head(10), x='Gym Name', y='Tailored Rating')
                        st.plotly_chart(fig)

                        # Option to download the recommendations as a CSV file
                        st.download_button(
                            label="Download Gym Recommendations as CSV",
                            data=sorted_gyms.to_csv(index=False),
                            file_name='tailored_gym_recommendations.csv',
                            mime='text/csv'
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

st.sidebar.title("About")
st.sidebar.info("This app helps you find gyms tailored to your preferences. Enter your location and rate the importance of different factors to get personalized recommendations.")
