# AutoPrice Insights

AutoPrice Insights is a web application designed to predict the prices of used cars. The project leverages HTML, CSS, Flask, and a Machine Learning model to provide accurate price predictions based on user input features.

## Features

- **User-Friendly Interface:** Built using HTML and CSS for an intuitive and responsive design.
- **Backend Framework:** Powered by Flask, providing a robust and scalable server-side application.
- **Machine Learning Model:** Employs a trained ML model to predict used car prices based on various input features.

## Installation

To run this project locally, follow these steps:

### Prerequisites

Ensure you have the following installed:

- Python 3.9+
- pip (Python package installer)
- Virtualenv (optional but recommended)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/govardhan-06/AutoPrice-Insights.git
cd AutoPrice-Insights
```

2. **Create and activate a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```

4. **Run the application:**

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000/`.

## Usage

1. Open your web browser and go to `http://127.0.0.1:5000/`.
2. Fill in the details of the car, such as make, model, year, mileage, and other relevant features.
3. Click on the 'Predict Price' button to get an estimated price for the used car.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## Acknowledgments

- The Flask framework (http://flask.pocoo.org/)
- Python and its scientific libraries (Pandas, NumPy, Scikit-learn)
- Bootstrap for front-end components (https://getbootstrap.com/)
