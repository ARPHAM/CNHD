# 7. TF-IDF (Term Frequency-Inverse Document Frequency)
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(tokens)])

# # Lấy danh sách các từ
# tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# # Lấy giá trị TF-IDF
# tfidf_scores = tfidf_matrix.toarray()[0]

# # In các từ cùng với giá trị TF-IDF
# print("TF-IDF scores:")
# for word, score in zip(tfidf_feature_names, tfidf_scores):
#     print(f"{word}: {score:.4f}")
