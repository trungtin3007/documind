import argparse
from retrieval import search_by_method, METHODS, METHOD_ALIASES, DEFAULT_METHOD, TOP_K

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the question to search for")
    ap.add_argument("--method", choices=list(METHODS) + list(METHOD_ALIASES), default=DEFAULT_METHOD)
    ap.add_argument("-k", type=int, default=TOP_K)
    args = ap.parse_args()

    question = " ".join(args.question) or input("Question: ")
    print(f"\nTop {args.k} pages for: {question!r}   [{args.method}]\n")
    for rank, (page, score) in enumerate(search_by_method(question, k=args.k, method=args.method), 1):
        print(f"{rank}. {page}   (score {score:.4f})")
