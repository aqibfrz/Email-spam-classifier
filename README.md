# Email/SMS Spam Classifier

A machine learning-based web application that classifies emails and SMS messages as either spam or ham (legitimate). Built with Streamlit and powered by Natural Language Processing (NLP) techniques.

## Features

- 🚀 **Real-time Classification**: Instantly classify emails and SMS messages as spam or ham
- 🎨 **User-friendly Interface**: Clean and intuitive web interface built with Streamlit
- 📊 **NLP-powered**: Uses advanced text preprocessing including tokenization, stopword removal, and stemming
- 🤖 **Machine Learning Model**: Pre-trained model for accurate spam detection
- 💻 **Easy to Use**: Simply paste your message and get instant results

## Project Structure

```
Email Spam Classifier/
│
├── app.py                      # Main Streamlit application
├── Spam_detection.ipynb        # Jupyter notebook for model training
├── model.pkl                   # Pre-trained machine learning model
├── vectorizer.pkl              # TF-IDF vectorizer for text preprocessing
├── spam.csv                    # Training dataset
├── pngwing.com (2).png         # Application logo/image
└── README.md                   # Project documentation
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository** (or download the project files)
   ```bash
   git clone <repository-url>
   cd "Email Spam Classifier"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install required dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn nltk pillow
   ```

5. **Download NLTK data**
   
   The application requires NLTK stopwords. Run Python and execute:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Usage

### Running the Application

1. **Activate your virtual environment** (if not already activated)

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser**
   
   The application will automatically open in your default web browser at `http://localhost:8501`

### How to Use

1. Enter or paste an email/SMS message in the text area
2. Click the **"Predict"** button
3. The application will display whether the message is **"Spam"** or **"ham"** (legitimate)

## How It Works

The spam classifier uses the following pipeline:

1. **Text Preprocessing**:
   - Converts text to lowercase
   - Tokenizes the text into individual words
   - Removes non-alphanumeric characters
   - Removes stopwords (common words like "the", "is", etc.)
   - Applies Porter Stemming to reduce words to their root form

2. **Feature Extraction**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert processed text into numerical features

3. **Classification**:
   - The pre-trained machine learning model predicts whether the message is spam (1) or ham (0)

## Model Training

The model was trained using the `Spam_detection.ipynb` Jupyter notebook. The training process includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Text processing and feature extraction
- Model building and evaluation
- Model serialization (saved as `model.pkl` and `vectorizer.pkl`)

To retrain the model, open the Jupyter notebook and run all cells.

## Dependencies

- **streamlit**: Web framework for building the user interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library (for model and vectorizer)
- **nltk**: Natural Language Processing toolkit
- **Pillow**: Image processing library

## Requirements

The project requires the following Python packages:

```
streamlit
pandas
numpy
scikit-learn
nltk
Pillow
```

To create a `requirements.txt` file, run:
```bash
pip freeze > requirements.txt
```

## Notes

- The model files (`model.pkl` and `vectorizer.pkl`) must be present in the project directory for the application to work
- The application uses a pre-trained model. For best results, ensure the model was trained on a diverse dataset
- The background image in the app is loaded from an external URL (Unsplash)

## License

This project is open source and available for educational purposes.

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## Author

Created as part of a machine learning project for email spam detection.

---

**Note**: This is a demonstration project. For production use, consider additional security measures, model validation, and performance optimization.

