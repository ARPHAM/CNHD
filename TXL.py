import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import ssl
from collections import Counter


# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context


# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# Đọc dữ liệu từ file .txt
with open('sample_text.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()


# 1. Chuyển thành chữ thường
text_data = text_data.lower()


# 2. Tokenization (Tách từ)
tokens = word_tokenize(text_data)


# 3. Loại bỏ từ dừng (Stopwords) và chuyển về thể gốc (Lemmatization)
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords.words('english')]


# BOW: Đếm tần số xuất hiện của các từ và sắp xếp các từ theo tần số xuất hiện
word_counts = Counter(tokens)
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_words:
    print(f"{word}: {count}")


# Sentiment Analysis: Phân tích cảm xúc với TextBlob
processed_text = ' '.join(tokens)
blob = TextBlob(processed_text)
sentiment = blob.sentiment
print(f"Phân tích cảm xúc: {sentiment}")
