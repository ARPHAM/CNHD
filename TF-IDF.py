import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Tải xuống dữ liệu cần thiết từ NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Tài liệu ví dụ
documents = [
    "I love programming in Python.",
    "Python programming is fun?",
    "I love learning new programming languages!"
]

# Lấy danh sách stop words tiếng Anh và dấu câu
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Tokenize và lọc bỏ stop words và dấu câu
filtered_documents = []

for doc in documents:
    tokens = word_tokenize(doc.lower())  # Tokenize và chuyển sang chữ thường
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in punctuation]
    filtered_documents.append(filtered_tokens)

# Hàm tính TF-IDF cho từng tài liệu
def compute_tfidf(documents):
    N = len(documents)  # Tổng số tài liệu
    tfidf_scores = []
    
    # Tính IDF cho toàn bộ tài liệu
    all_tokens = set(token for doc in documents for token in doc)
    idf_dict = {}
    
    for term in all_tokens:
        containing_docs = sum(1 for doc in documents if term in doc)
        idf_dict[term] = math.log((N) / (containing_docs))  # IDF với làm trơn

    # Tính TF-IDF cho từng tài liệu
    for tokens in documents:
        tf_dict = {}
        total_terms = len(tokens)
        
        # Tính TF cho từng từ trong tài liệu
        for term in tokens:
            tf_dict[term] = tf_dict.get(term, 0) + 1
            
        # Chuẩn hóa TF
        for term in tf_dict:
            tf_dict[term] /= total_terms
        
        # Tính TF-IDF
        tfidf = {term: tf_dict[term] * idf_dict[term] for term in tf_dict}
        tfidf_scores.append(tfidf)
        
    return tfidf_scores

# Tính TF-IDF cho từng tài liệu
tfidf_scores = compute_tfidf(filtered_documents)

# Hiển thị các giá trị TF-IDF cho từng tài liệu
for i, doc in enumerate(tfidf_scores):
    print(f"\nDocument {i + 1} TF-IDF Scores:")
    for term, score in doc.items():
        print(f"{term}: {score:.4f}")
