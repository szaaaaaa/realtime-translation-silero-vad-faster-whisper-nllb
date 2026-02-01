import shutil
import os

def zip_app():
    source_dir = os.path.join('dist', 'MeetingTranslator')
    output_filename = 'MeetingTranslator'
    
    print(f"Zipping {source_dir} to {output_filename}.zip...")
    
    archive_path = shutil.make_archive(output_filename, 'zip', root_dir='dist', base_dir='MeetingTranslator')
    print(f"Zip created successfully at: {archive_path}")
    
    print("Current directory contents:")
    print(os.listdir('.'))

if __name__ == "__main__":
    zip_app()
