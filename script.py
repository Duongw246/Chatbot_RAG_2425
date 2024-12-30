import doc2docx
import os
import shutil
from glob import glob

# Define paths
source_dir = r"D:\Documents\TDTU\ProjectIT\Chatbot_RAG_2024\data\unprocessed_documents"
target_dir = r"D:\Documents\TDTU\ProjectIT\Chatbot_RAG_2024\data\documents"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Get all .doc and .docx files in the source directory
doc_files = glob(os.path.join(source_dir, "*.doc"))
docx_files = glob(os.path.join(source_dir, "*.docx"))

# Function to convert .doc files to .docx
def convert(file):
    try:
        doc2docx.convert(file)
        print(f"Converted: {file}")
    except Exception as e:
        print(f"Failed to convert {file}: {e}")

# Function to move .docx files to the target directory
def move_files(files):
    for file in files:
        try:
            file_name = os.path.basename(file)  # Extract file name
            target_file = os.path.join(target_dir, file_name)
            shutil.move(file, target_file)
            print(f"Moved: {file_name}")
        except Exception as e:
            print(f"Failed to move {file}: {e}")

# Convert .doc files to .docx
for doc_file in doc_files:
    convert(doc_file)

# Refresh .docx files list to include newly converted files
docx_files = glob(os.path.join(source_dir, "*.docx"))

# Move all .docx files
move_files(docx_files)

print("All tasks completed successfully!")
