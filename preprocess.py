import csv
import os
import pickle
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def parse_stanford_courses(file_path):
    courses = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in tqdm(reader, desc="Parsing CSV"):
            courses.append(row)
    return courses

def create_course_chunks(course):
    course_text = f"""
    Course ID: {course['course_id']}
    Title: {course['title']}
    Description: {course['description']}
    Units: {course['units_min']} - {course['units_max']}
    GERs: {course['gers']}
    Academic Group: {course['academic_group']}
    Grading: {course['grading_basis']}
    Repeatable: {course['repeatable']}
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(course_text)
    
    metadata = {
        "course_id": course["course_id"],
        "title": course["title"],
        "academic_group": course["academic_group"],
        "units_min": course["units_min"],
        "units_max": course["units_max"],
        "gers": course["gers"],
        "grading_basis": course["grading_basis"],
        "repeatable": course["repeatable"]
    }
    
    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, '..', 'data', 'stanford_courses.csv')
index_path = os.path.join(current_dir, '..', 'server', 'faiss_index.pickle')
metadata_path = os.path.join(current_dir, '..', 'server', 'metadata.pkl')
retriever_path = os.path.join(current_dir, '..', 'server', 'retriever_components.pkl')

csv_file_path = os.path.abspath(csv_file_path)
index_path = os.path.abspath(index_path)
metadata_path = os.path.abspath(metadata_path)
retriever_path = os.path.abspath(retriever_path)

print(f"CSV file path: {csv_file_path}")
print(f"Index path: {index_path}")
print(f"Metadata path: {metadata_path}")
print(f"Retriever path: {retriever_path}")

print("Parsing Stanford courses...")
courses = parse_stanford_courses(csv_file_path)

print("Creating document chunks...")
docs = []
for course in tqdm(courses, desc="Creating document chunks"):
    docs.extend(create_course_chunks(course))

print(f"Total chunks created: {len(docs)}")

print("Creating embeddings and FAISS index (this may take a while)...")
embeddings = OpenAIEmbeddings()

print("Creating FAISS index...")
vectorstore = FAISS.from_documents(docs, embeddings)

print("Saving index, metadata, and retriever components...")
vectorstore.save_local(index_path)

with open(metadata_path, 'wb') as f:
    pickle.dump({"docs": docs}, f)

retriever_components = {
    "bm25_docs": docs,
}
with open(retriever_path, 'wb') as f:
    pickle.dump(retriever_components, f)

print(f"Number of documents processed: {len(docs)}")
print(f"Sample text: {docs[0].page_content[:100]}...")  
print(f"Sample metadata: {docs[0].metadata}")
print("Index, metadata, and retriever components saved successfully.")
print("Preprocessing complete.")