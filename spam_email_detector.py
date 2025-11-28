# spam_email_detector.py
# Basic AIML-based Spam Email Detector
# Suitable for B.Tech CSE 1st year project

# === 1. IMPORTS ===
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# === 2. SAMPLE DATASET (YOU CAN EXPAND THIS) ===
# emails: list of email texts (strings)
emails = [
    "Congratulations! You have won a lottery of $1,000,000. Click here to claim now.",
    "Dear user, your account has been suspended. Click the link to verify your details.",
    "Limited time offer!!! Buy now and get 70% discount on all products.",
    "You have been selected for a free gift voucher. Reply with your bank details.",
    "Earn money fast from home. No experience required. Register today.",
    "Your OTP for login is 458921. Do not share it with anyone.",
    "Hi Mom, I will be coming home this weekend. Please prepare my favourite food.",
    "Meeting is scheduled tomorrow at 10 AM. Please join on time.",
    "Your order has been shipped and will be delivered by Monday.",
    "Can we reschedule our project discussion to next Tuesday?",
    "Reminder: Your electricity bill is due tomorrow. Please make the payment.",
    "Let’s catch up this evening for coffee if you are free.",
]

# labels: corresponding labels for each email
# "spam" = spam email, "ham" = normal email
labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "spam",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
    "ham",
]

# === 3. CONVERT TEXT TO NUMERICAL FEATURES ===
# CountVectorizer converts text into a matrix of word counts
vectorizer = CountVectorizer()

# Learn vocabulary and transform emails into feature vectors
X = vectorizer.fit_transform(emails)  # Features
y = labels                            # Targets

# === 4. SPLIT INTO TRAINING AND TESTING SETS ===
# test_size=0.25 means 25% data will be used for testing
# random_state is for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# === 5. CHOOSE AND TRAIN THE MODEL ===
# Multinomial Naive Bayes is commonly used for text classification
model = MultinomialNB()
model.fit(X_train, y_train)

# === 6. EVALUATE THE MODEL ===
y_pred = model.predict(X_test)

print("=== MODEL EVALUATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# === 7. INTERACTIVE PREDICTION ===
def predict_email(email_text: str) -> str:
    """
    Takes a raw email text (string),
    converts it using the same vectorizer,
    and returns 'spam' or 'ham'.
    """
    # Transform the input email using the already fitted vectorizer
    email_features = vectorizer.transform([email_text])

    # Predict using the trained model
    prediction = model.predict(email_features)[0]
    return prediction

print("\n=== SPAM EMAIL DETECTOR ===")
print("Type an email content to check if it is SPAM or NOT SPAM.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter email text: ")

    if user_input.lower().strip() == "exit":
        print("Exiting Spam Email Detector. Goodbye!")
        break

    result = predict_email(user_input)

    if result == "spam":
        print("👉 Prediction: This email is likely SPAM.\n")
    else:
        print("✅ Prediction: This email is NOT SPAM (HAM).\n")
