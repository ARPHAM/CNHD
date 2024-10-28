import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import ssl
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context

# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Đọc dữ liệu từ file .txt
with open('sample_text.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# 1. Chuyển thành chữ thường
text_data = text_data.lower()
print(f"Văn bản chữ thường: {text_data}")

# 2. Tokenization (Tách từ)
tokens = word_tokenize(text_data)
print(f"Văn bản sau khi tách: {tokens}")

# 3. Loại bỏ từ dừng (Stopwords) và chuyển về thể gốc (Lemmatization)
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords.words('english')]
print(f"Văn bản loại bỏ từ dừng và chuyển về thể gốc: {tokens}")

# 4. BOW: Đếm tần số xuất hiện của các từ và sắp xếp các từ theo tần số xuất hiện
word_counts = Counter(tokens)
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
print("Từ và tần số xuất hiện:")
for word, count in sorted_words:
    print(f"{word}: {count}")

# 5. Sentiment Analysis: Phân tích cảm xúc với TextBlob
processed_text = ' '.join(tokens)
blob = TextBlob(processed_text)
sentiment = blob.sentiment
print(f"Phân tích cảm xúc: {sentiment}")

# 6. N-gram (bigram và trigram)
def generate_ngrams(tokens, n):
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

bigrams = generate_ngrams(tokens, 2)
trigrams = generate_ngrams(tokens, 3)

print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")

# 7. TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens)])

# Lấy danh sách các từ
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# Lấy giá trị TF-IDF
tfidf_scores = tfidf_matrix.toarray()[0]

# In các từ cùng với giá trị TF-IDF
print("TF-IDF scores:")
for word, score in zip(tfidf_feature_names, tfidf_scores):
    print(f"{word}: {score:.4f}")
