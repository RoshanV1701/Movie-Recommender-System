# Movie Recommender System

A web-based movie recommendation engine utilizing ML. The system offers personalized movie recommendations by identifying patterns of use between similar users and their preferences. If two users like similar movies, it suggests movies one user has watched and enjoyed to the other, enhancing the discovery of new favorites.

## Key Features:

- **Home Page**: A welcoming interface where users can start their movie discovery journey.
  
- **Recommendation Page**: Displays personalized movie suggestions based on user preferences and similarities with others.
  
- **Rating Page**: Allows users to rate movies, improving the system's recommendation accuracy and also displays similar movies to the given movie.

## Technologies Used:

- **Web Technologies**: HTML, CSS, JavaScript, Bootstrap, Django
- **Machine Learning Libraries (Python 3)**: Numpy, Pandas, Scipy, SkLearn
- **Database**: SQLite

## Setup Requirements:

- Python 3.9
- pip3
- virtualenv

## Installation and Local Server Setup:

1. **Extract** the zip file to your computer.
2. **Open** a terminal or command prompt.
3. **Navigate** to the extracted folder
4. **Create** a virtual environment (`virtualenv .`).
5. **Activate** the virtual environment:
   - Linux: `source bin/activate`
   - Windows: `Scripts\activate`
6. **Install Dependencies**: `pip install -r requirements.txt`.
7. **Launch** the application:
   - Navigate to the `src` directory (`cd src`).
   - Run the server (`python manage.py runserver`).
8. **Access** the application in your browser at `http://127.0.0.1:8000`.

Thank you for exploring the Movie Recommender System. Dive into a personalized movie discovery experience powered by advanced machine learning algorithms.
