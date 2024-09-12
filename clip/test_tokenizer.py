# simple_tokenizer.py에서 SimpleTokenizer 클래스 가져오기
from simple_tokenizer import SimpleTokenizer

if __name__ == "__main__":
    tokenizer = SimpleTokenizer()  # 토크나이저 초기화
    test_text = "This is an example of BPE tokenization."
    
    # 인코딩
    encoded_tokens = tokenizer.encode(test_text)
    print(f"Encoded Tokens: {encoded_tokens}")
    
    # 디코딩
    decoded_text = tokenizer.decode(encoded_tokens)
    print(f"Decoded Text: {decoded_text}")
