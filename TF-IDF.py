from sklearn.feature_extraction.text import TfidfVectorizer

# Danh sách các văn bản
documents = [
    "Học máy là một lĩnh vực của trí tuệ nhân tạo",
    "Trí tuệ nhân tạo là tương lai của công nghệ",
    "Python là ngôn ngữ lập trình phổ biến cho học máy"
]

# Khởi tạo bộ vector TF-IDF
vectorizer = TfidfVectorizer()

# Tính toán TF-IDF
tfidf_matrix = vectorizer.fit_transform(documents)

# Hiển thị kết quả
print("Các từ trong từ điển: ", vectorizer.get_feature_names_out())
print("Ma trận TF-IDF:\n", tfidf_matrix.toarray())
