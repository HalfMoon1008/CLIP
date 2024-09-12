## 목표 : SimpleTokenizer라는 함수를 통해 BPE(Basic Pair Encoding) 방식으로 txt를 처리
# To do : 불용어처리, 인코딩을 통해 BPE 방식으로 토큰화
# 해당 토큰을 다시 텍스트로 디코딩

## 기본 구조
# 전처리 : 불용어 처리 등 텍스트를 정리, 바이트와 유니코드로 변환하는 기능
# BPE 인코딩 & 디코딩 : BPE 규칙에 따라 텍스트를 처리

# 기본 동작을 위한 라이브러리
import os
import gzip #gz로 압축돼있는 bpe 규칙파일을 위해 사용
from functools import lru_cache #캐싱을 위한
import html

# 전처리를 위한 라이브러리
import ftfy #pip install ftfy, 깨진 문자 자동 수정
import regex as re



@lru_cache() #함수의 결과를 캐싱
# BPE 규칙 파일 읽기
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),"bpe_simple_vocab_16e6.txt.gz")
    # dirname : 경로명 path의 디렉터리 이름을 반환
    # os.path.abspath(__file__) : 현재 코드를 실행하는 파일의 절대경로를 반환
    # 현재의 dir의 절대경로를 이용하여 bpe 규칙 파일을 넣겠다.

@lru_cache()
def bytes_to_unicode():
    """
    UTF-8 바이트와 이에 대응하는 유니코드 문자열을 반환하는 함수
    BPE 코드는 유니코드 문자열에서 작동하기 때문에 이 함수로 바이트와 유니코드를 매핑하여
    BPE로 인코딩 되도록 도와줌
    """
    ## 앵간한 모든 문자, 기호, 숫자 등을 인코딩하기 위한 리스트
    bs = (
    list(range(ord("!"), ord("~")+1)) +  # 33번부터 126번까지 (ASCII 특수문자, 숫자, 대소문자 포함)
    list(range(ord("¡"), ord("¬")+1)) +  # 161번부터 172번까지 (유니코드 라틴 보충 기호)
    list(range(ord("®"), ord("ÿ")+1))    # 174번부터 255번까지 (유니코드 라틴 보충 기호)
    )
    # ord는 문자를 유니코드 코드 포인트(정수 값)으로 변환해줌
    # ord('A') = 65 반환

    cs = bs[:]
    n = 0

    # 256개의 바이트 값 중 기존에 정의되지 않은 값들을 추가
    for b in range(2**8): #2^8
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            # 기존의 유니코드 값(bs)과 겹치지 않도록 새로운 유니코드 값(256부터 시작하는)을 사용
            n += 1
    cs = [chr(n) for n in cs]  # 해당 값을 유니코드로 변환
    # bs는 바이코드 값을 저장
    # cs는 bs에 대응하는 유니코드 값을 저장 (매핑을 위한 변수)
    # ex) bs[i]와 cs[i]가 매칭
    return dict(zip(bs, cs))  # 바이트와 유니코드를 매핑한 딕셔너리를 반환

def get_pairs(word):
    """
    주어진 단어에서 연속된 문자 쌍의 집합을 반환하는 함수
    """
    pairs = set()
    prev_char = word[0]
    # 문자들을 순회하면서 문자 쌍을 만들어 집합에 추가
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs  # 문자 쌍의 집합을 반환
# ex) apple
# pairs = ('a', 'p'), ('p', 'p'), ('p', 'l'), ('l', 'e')

def basic_clean(text):
    """
    깨진 텍스트 수정: 잘못된 인코딩이나 깨진 문자를 수정
    HTML 엔티티 해제: HTML 코드에서 사용되는 특수 문자(예: &amp; → &)를 원래의 문자로 변환
    불필요한 공백 제거: 앞뒤에 존재하는 불필요한 공백을 제거하여 깨끗한 텍스트 반환
    """
    # ftfy는 깨진 인코딩을 자동으로 수정해주는 라이브러리
    # 텍스트가 "Ã©" 같은 잘못된 문자를 포함하고 있다면, 이를 "é"로 수정
    # 특수 문자를 표현하는 HTML 코드(HTML 엔티티)를 원래 문자로 변환

    text = ftfy.fix_text(text)  # 텍스트에서 깨진 부분을 수정
    text = html.unescape(html.unescape(text))  # HTML 엔티티를 디코딩
    return text.strip()  # 앞뒤 공백을 제거한 텍스트를 반환

## 텍스트에서 여러 개의 공백을 하나로 줄이고, 앞뒤 공백을 제거하는 함수
def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    # 생성자 함수: 클래스가 호출될 때 실행되며 BPE 파일 경로를 초기화
    def __init__(self,bpe_path:str = default_bpe()):

        self.byte_encoder = bytes_to_unicode() # 바이트 -> 유니코드 매핑을 생성
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 유니코드 -> 바이트 매핑을 생성
        # byte_encoder를 역방향으로 변환

        try:
            with gzip.open(bpe_path, 'r') as f:
                merges = f.read().decode("utf-8").splitlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"BPE 규칙 파일을 찾을 수 없음: {bpe_path}")

        # BPE 병합 규칙 파일을 읽음
        # 압축된 BPE 파일을 열고, UTF-8로 디코딩 후 줄 단위로 분리
        merges = merges[1:] # BPE 파일의 첫 번째 줄은 무시 (보통 첫 줄은 메타 데이터, 나머지 병합 규칙만 리스트에 저장)
        merges = [tuple(merge.split()) for merge in merges]  # 각 줄을 튜플로 변환. 규칙을 불변의 형태로 진화!
        self.bpe_ranks = dict(zip(merges, range(len(merges))))  # BPE 병합 규칙을 순위로 변환, BPE 병합을 수행할 때 우선순위를 결정하는 데 사용
        self.cache = {}  # 토큰화 결과를 캐싱하기 위한 딕셔너리
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+""", re.IGNORECASE)  # 텍스트에서 단어, 숫자, 특수문자 등을 추출

        vocab = list(bytes_to_unicode().values())  # 바이트 -> 유니코드 변환값 리스트로 생성
        vocab = vocab + [v+'</w>' for v in vocab]  # 각 유니코드 값에 '</w>'를 추가해 단어의 끝을 표시
        for merge in merges:
            vocab.append(''.join(merge))  # 병합된 문자쌍을 어휘에 추가
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])  # 시작 및 끝 태그 추가
        self.encoder = dict(zip(vocab, range(len(vocab))))  # 어휘를 인코딩할 인코더 생성
        self.decoder = {v: k for k, v in self.encoder.items()}  # 인코더의 역매핑을 사용해 디코더 생성

    def bpe(self, token):
        """
        주어진 토큰을 BPE(Byte Pair Encoding) 방식으로 인코딩하는 함수
        - BPE는 자주 등장하는 문자 쌍을 병합하여 단어를 효율적으로 인코딩하는 방식
        - 주어진 토큰을 BPE 병합 규칙에 따라 점진적으로 병합하고, 결과를 반환
        """
        # 캐시 확인
        if token in self.cache:
            return self.cache[token]
        # 토큰을 투플로 변환, 마지막에 /w추가
        # ex) "apple" -> ('a', 'p', 'p', 'l', 'e</w>')
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        # 문자 쌍 추출
        pairs = get_pairs(word)

        # 단일 문자라면 /w추가
        if not pairs:
            return token+'</w>'

        # 슈슈슉 병합!
        # 병합할 쌍이 있으면 변환 ㄱㄱ
        # self.bpe.ranks를 이용하여 각 문자쌍의 병합 순위를 기준으로 병합
        while True:
            # 현재 존재하는 문자 쌍 중에서 BPE 병합 규칙에 따라 가장 낮은 순위를 가진 문자 쌍을 선택
            # self.bpe_ranks.get(pair, float('inf'))는 순위를 확인하고, 없는 쌍은 무한대 값(float('inf'))으로 처리
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            # 병합할 문자 쌍을 분리 (예: bigram = ('p', 'p'))
            first, second = bigram
            new_word = []  # 병합 후 새로운 단어를 저장할 리스트
            i = 0  # 단어의 첫 문자부터 순차적으로 병합

            # 현재 단어를 순회하며 문자 쌍을 찾아 병합
            while i < len(word):
                try:
                    # 현재 단어에서 병합할 첫 번째 문자가 있는 위치를 찾음
                    j = word.index(first, i)
                    # 첫 번째 문자가 나오기 전까지의 문자를 새로운 단어 리스트에 추가
                    new_word.extend(word[i:j])
                    i = j
                except:
                    # 더 이상 찾을 수 없으면 나머지 문자를 모두 추가하고 종료
                    new_word.extend(word[i:])
                    break

                # 첫 번째 문자가 현재 위치에 있고, 다음 문자가 병합할 두 번째 문자라면 병합
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)  # 두 문자를 합쳐 병합
                    i += 2  # 병합된 두 문자를 건너뜀
                else:
                    new_word.append(word[i])  # 병합되지 않으면 현재 문자를 그대로 추가
                    i += 1

            # 새로운 단어 리스트를 튜플로 변환 (튜플을 사용하는 이유는 불변성 때문)
            new_word = tuple(new_word)
            word = new_word  # 새로 병합된 단어를 업데이트

            # 병합 후 단어가 더 이상 병합할 수 없는 하나의 요소로만 이루어지면 종료
            if len(word) == 1:
                break
            else:
                # 병합할 새로운 문자 쌍을 다시 구함
                pairs = get_pairs(word)

        # 최종 병합된 단어를 공백으로 구분된 문자열로 변환
        word = ' '.join(word)
        # 캐시에 결과를 저장하여, 동일한 토큰에 대해 다시 계산하지 않도록 함
        self.cache[token] = word
        return word  # 최종 결과 반환

    
    def encode(self, text):
        """
        주어진 텍스트 -> BPE 토큰으로 인코딩하는 함수
        텍스트를 정리하고, 각 단어를 BPE 방식으로 인코딩하여 BPE 토큰으로 변환
        """
        bpe_tokens = []  # BPE로 변환된 토큰을 저장할 리스트

        # 기본적인 텍스트 정리: 불필요한 HTML 엔티티를 제거하고 공백을 정리한 후 소문자로 변환
        text = whitespace_clean(basic_clean(text)).lower()

        # 정규 표현식 패턴을 사용해 텍스트를 토큰으로 분리 (단어, 숫자, 특수 문자 등)
        # self.pat는 토큰화에 사용되는 정규 표현식 패턴을 정의
        for token in re.findall(self.pat, text):
            # 각 토큰을 바이트 단위로 인코딩하고, 이를 유니코드 값으로 변환 (바이트 -> 유니코드 매핑 사용)
            try:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            except KeyError as e:
                print(f"Encoding error : {e}")
                return []
            # 변환된 바이트 토큰을 BPE 방식으로 인코딩하고, 결과를 BPE 토큰 리스트에 추가
            # BPE 방식으로 토큰을 병합하여 공백으로 구분된 BPE 토큰 리스트로 변환
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))

        return bpe_tokens  # 최종적으로 BPE로 인코딩된 토큰 리스트를 반환

    
    def decode(self, tokens):
        """
        BPE 토큰 -> 텍스트로 디코딩하는 함수
        BPE 방식으로 인코딩된 토큰을 원래의 텍스트로 복원
        """
        # 각 BPE 토큰을 유니코드 값에서 문자로 변환 (BPE 토큰 -> 문자로 변환)
        text = ''.join([self.decoder[token] for token in tokens])

        # 문자들을 다시 바이트 값으로 디코딩하여 원래의 텍스트로 복원
        # bytearray로 바이트 단위로 변환 후, 유니코드 문자로 디코딩
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        
        # 3. 단어 끝을 나타내는 </w> 태그를 공백으로 대체하여 최종 텍스트를 반환
        return text  # 최종 복원된 텍스트 반환
    

