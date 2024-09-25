# 이 파일은 서비스 배포에 치중된 파일...!
# 당장의 연구에서 이 정도까지 할 필요는 없을 듯 하다..!
# 아키텍쳐 구현을 좀 더 봐야할 듯 함...!

## Q) = 좀 더 알아보기

from .model import build_model # 모델을 생성하는 함수
from .simple_tokenizer import SimpleTokenizer as _Tokenizer


import hashlib  # 해시 함수를 사용하기 위한 라이브러리 (파일의 무결성을 확인할 때 사용됨)
import os  
import urllib  # URL에서 파일을 다운로드할 때 사용하는 라이브러리
import warnings  # 경고 메시지를 표시하기 위한 라이브러리 (특정 조건에서 경고를 줄 때 사용됨)
from packaging import version  # 버전 비교를 위한 라이브러리 (소프트웨어의 버전을 비교할 때 사용됨)
from typing import Union, List  # 타입 힌트(코드 가독성을 높이고 오류를 줄여줌)를 위한 라이브러리, __all__ Q) 형변환 자세하게 알아보자

import torch  
from PIL import Image  # PIL은 이미지를 처리하고 조작하기 위한 라이브러리임
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  # 이미지 변환(크기 조정, 자르기, 텐서로 변환 등)에 필요한 도구들을 불러옴
from tqdm import tqdm  

# 이미지 보간(interpolation) 방법을 설정함. 보간은 이미지 크기를 조정할 때 픽셀 값을 계산하는 방법
# 만약 최신 버전의 'InterpolationMode'가 있으면 그걸 사용하고, 없으면 기존의 'BICUBIC'을 사용
# interpolationmode : 기존의 픽셀값을 기준으로 새로운 픽셀을 추정
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC  # 이미지 품질이 좋은 BICUBIC 방식을 사용
except ImportError:
    BICUBIC = Image.BICUBIC  # 'InterpolationMode'가 없으면 PIL의 BICUBIC을 사용

# PyTorch의 버전이 1.7.1보다 낮으면 경고 메시지를 출력
if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

# 이 모듈에서 사용할 수 있는 함수들을 미리 지정
__all__ = ["available_models", "load", "tokenize"]
# available_models: 사용 가능한 모델 목록을 반환하는 역할
# 모듈을 사용하는 사람들에게 어떤 모델들이 사용 가능한지를 알려주기 때문에, 외부에서 이 정보를 얻는 것이 매우 유용

# load: 모델을 로드하는 핵심 함수
# 모델을 불러와서 실제로 사용하려면 외부에서 이 함수에 접근할 수 있어야 함

# tokenize: 문자열(텍스트)을 토큰화하여 모델이 처리할 수 있는 형태로 변환하는 역할
# 텍스트 데이터를 입력으로 사용할 때 필수적인 전처리 과정이기 때문

# ex) def available_models() -> List[str]:
# from typing import Union, List을 사용해서 위와 같이 함수를 정의 할 수 있다.
# 형 변환이 편해짐
# Q)

# 텍스트를 토큰화하는 데 사용할 토크나이저 객체를 생성
_tokenizer = _Tokenizer()

# 사용 가능한 모델 목록을 저장, 각 모델의 이름과 모델을 다운로드할 수 있는 URL이 매핑되어 있음
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

# 파일을 다운로드하고, 다운로드한 파일의 해시 값을 검사하여 파일이 손상되지 않았는지 확인하는 함수
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)  # 파일이 저장될 디렉토리가 없으면 새로 만듦
    filename = os.path.basename(url)  # URL에서 파일 이름만 추출

    expected_sha256 = url.split("/")[-2]  # URL에서 예상되는 해시 값을 추출, sha256 : URL(문자열)을 가져올 수 있는 해시 알고리즘 Q)
    download_target = os.path.join(root, filename)  # 파일이 저장될 경로를 설정

    # 파일이 이미 존재하지만 정상적인 파일이 아니면 오류를 발생
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 파일이 이미 존재하고, 해시 값이 맞으면 그대로 반환함 (재다운로드하지 않음)
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # URL에서 파일을 다운로드
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 파일을 다운로드하는 동안 진행률을 표시
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop: # iB : 이진바이트
            while True:
                buffer = source.read(8192)  # 8192 byte씩 데이터를 읽음 _ A) 8192는 파일을 다운로드하거나 복사할 때 효율적인 데이터 전송을 위해 자주 사용되는 버퍼 크기 (성능 최적화, 표준 값)
                if not buffer:
                    break  # 더 이상 읽을 데이터가 없으면 종료함

                output.write(buffer)  # 읽은 데이터를 파일에 씀
                loop.update(len(buffer))  # 진행률을 업데이트

    # 다운로드한 파일의 해시 값을 확인
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target  # 다운로드한 파일 경로를 반환

# 이미지를 RGB 형식으로 변환하는 함수
def _convert_image_to_rgb(image):
    return image.convert("RGB")

# 이미지를 전처리(크기 조정, 자르기, 색상 변환 등)하는 함수
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),  # 크기를 n_px로 조정함
        CenterCrop(n_px),  # 이미지를 중앙에서 잘라냄
        _convert_image_to_rgb,  # 이미지를 RGB로 변환
        ToTensor(),  # 이미지를 텐서로 변환 (딥러닝 모델에서 입력으로 사용 가능하게 함)
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # 이미지의 각 채널을 정규화
    ])

# 사용 가능한 모델 목록을 반환하는 함수
def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())  # 모델의 이름들을 리스트로 반환

# CLIP 모델을 불러오는 함수
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    # JIT : Just-In-Time, 사전에 기계어로 모두 번역하는 컴파일이 아닌 line by line으로 필요한 부분만 컴파일
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # 모델 이름이 목록에 있으면 해당 모델을 다운로드
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name  # 모델이 이미 파일로 있으면 그 경로를 사용
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # JIT 모델을 로드하려고 시도
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # JIT 모델 로드가 실패하면 일반적인 상태 사전을 로드
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    # JIT가 꺼져 있으면 모델을 빌드하고 장치에 올림
    # Q) model.state_dict()에 대해 좀 더 알아보기
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":  # CPU에 모델을 올렸으면 float 형식으로 변환
            model.float() # model.state_dict()로 하면 K와 V의 dict형태로 저장되는데, 이때 V가 tensor값이기 때문에 float로 변환 됨
        return model, _transform(model.visual.input_resolution) # 전처리 파이프라인에 해당 Q)

    # 장치 이름을 패치
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]) # 내 코드가 돌아가는 환경이 CPU인지, GPU인지 추적하는 코드
        # torch.jit.trace : 주어진 함수의 실행을 추적하고, 그 함수가 실행되는 과정을 기록하여 그래프 형태로 저장
        # torch.one([]) : 크키가 없는 텐서를 만듬. 이 텐서는 device 정보를 설정하는데 사용
        # .to(torch.device(device))를 통해 텐서를 지정된 장치로 이동시킴, device = 'cuda'면 텐서가 GPU로, device = 'cpu'면 텐서가 CPU로 이동함
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]
        # prim::Constant : JIT 컴파일 시 장치 정보와 관련된 상수를 나타냄
        # [-1]로 마지막 장치 정보에 대한 정보를 tensor에 저장
        # 즉, JIT 모델 내 장치 설정을 반영해서 다른 사람이 이 코드를 사용하더라도 의도된 장비(GPU or CPU etc..)에서 코드가 돌아가도록 확인시키는 코드

        # JIT는 코드를 작성할 때 부터 들어가있는 코드 작성자의 배려로 느껴짐
        # "이 코드를 clone해가서 쓸 당신들의 환경에서 편의를 좀 더 제공하고자 내가 설정을 미리 해줄게"
    

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key) # kindOf : (key)의 타입을 반환. 정수, 문자열, 장치 정보 등등
        return getattr(node, sel)(key) # getattr : 동적으로 객체의 속성을 가져오는 함수
    
        # 모델이 GPU에서 실행되도록 설정되어 있는지 확인하고 싶을 때, "Device"라는 키를 사용하여 해당 속성을 가져옴
        # 속성의 반환값은 'cuda' or 'cpu' 등이 됨

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else [] 
            # module.graph: module 객체가 JIT 컴파일된 모델일 경우, 그 모델은 그래프 형태로 컴파일
            # 그래프는 모델의 연산을 나타내며, 모든 연산과 관련된 노드들을 포함
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
            # 일부 모듈은 'forward1' 이라는 또 다른 그래프를 가질 수 있고, 그래프에 추가함

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)
                    # 어떤 deive든! node의 형태로 model.dict()에 있는 value들을 가져오겠다.

        # JIT 컴파일된 모델이 실행되는 동안 장치 정보(CPU 또는 GPU)가 잘못 설정되거나, 시스템 환경에 맞지 않게 설정될 경우 모델이 제대로 작동하지 않을 수 있음
        # 이를 방지하기 위해 장치 정보를 추적하고, 필요시 수정(forward1을 사용해서)하여 모델이 의도한 대로 실행되도록 보장하는 역할

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # CPU에서 dtype을 float32로 패치
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype이 두 번째 또는 세 번째 인자로 전달될 수 있음
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

# 문자열(또는 문자열 목록)을 토큰화하여 텐서로 변환하는 함수
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    # 입력이 단일 문자열이면 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    # 시작 토큰과 종료 토큰을 설정
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    # 모든 문자열을 토큰화하고 시작/종료 토큰을 추가
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # PyTorch 버전에 따라 텐서 자료형을 다르게 설정
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)  # 오래된 버전에서는 long 타입 사용
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)  # 최신 버전에서는 int 타입 사용

    # 각 텍스트에 대해 토큰을 텐서로 변환
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:  # 토큰이 너무 길면
            if truncate:  # 자르기 옵션이 켜져 있으면 자르고 종료 토큰을 붙임
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)  # 텐서에 토큰 값을 채워 넣음

    return result  # 최종적으로 토큰화된 텐서를 반환
