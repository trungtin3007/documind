import sys
from retrieval import search, TOP_K

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    print(f"\nTop {TOP_K} pages for: {question!r}\n")
    for rank, (page, score) in enumerate(search(question), 1):
        print(f"{rank}. {page}   (score {score:.3f})")
