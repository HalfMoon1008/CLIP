# CLIP

```plaintext
CLIP/
├── clip/
│   ├── clip.py                  # CLIP 모델의 핵심 구현 파일로, 모델 초기화 및 추론 과정이 포함
│   ├── model.py                 # CLIP 모델의 세부적인 모델 구조를 정의한 파일
│   ├── simple_tokenizer.py       # 간단한 토큰화 도구를 정의한 파일로, 입력 텍스트를 토큰으로 변환하는 역할
│   └── test_tokenizer.py         # 토크나이저의 동작을 테스트하는 코드 파일
├── test/
│   └── test_consistency.py       # CLIP 모델의 일관성을 확인하는 테스트 스크립트
```

---

Original source: https://github.com/openai/CLIP
