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

# Hàm để xử lý văn bản: chuyển thành chữ thường, loại bỏ từ dừng và lemmatization
def process_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return ' '.join(tokens)

# Đọc dữ liệu từ hai file .txt
file1 = 'sample_text.txt'
file2 = 'sample_text2.txt'

corpus = []

# Xử lý file 1
with open(file1, 'r', encoding='utf-8') as f1:
    text_data1 = f1.read()
    processed_text1 = process_text(text_data1)
    corpus.append(processed_text1)

# Xử lý file 2
with open(file2, 'r', encoding='utf-8') as f2:
    text_data2 = f2.read()
    processed_text2 = process_text(text_data2)
    corpus.append(processed_text2)

# Tính TF-IDF cho cả hai tài liệu
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Lấy danh sách các từ
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# In giá trị TF-IDF cho từng tài liệu
for doc_idx, doc in enumerate(corpus):
    print(f"\nTF-IDF scores for document {doc_idx + 1}:")
    tfidf_scores = tfidf_matrix[doc_idx].toarray()[0]
    for word, score in zip(tfidf_feature_names, tfidf_scores):
        if score > 0:  # Chỉ in những từ có điểm TF-IDF khác 0
            print(f"{word}: {score:.4f}")
