import nltk
import subprocess



if __name__ == "__main__":
    try:
        subprocess.run(['python', '-m', 'unidic', 'download'], check=True)
        print("Successfully download UNIDIC")
    except subprocess.CalledProcessError as e:
        print(f"Fail to download UNIDIC: {e}")
        
    nltk.download("punkt")
    nltk.download("punkt_tab")