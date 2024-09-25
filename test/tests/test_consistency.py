import numpy as np
import pytest  # 테스트를 위한 프레임워크 (여기서는 파라미터화된 테스트를 위해 사용됨)
import torch
from PIL import Image
import clip  # CLIP 모델을 사용하기 위한 라이브러리 (텍스트와 이미지를 함께 처리)

# 'pytest.mark.parametrize' 데코레이터를 사용하여 여러 모델에 대해 테스트를 실행
# 'clip.available_models()' 함수로 사용 가능한 모든 모델을 가져옴
@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):
    """
    CLIP 모델의 JIT 모드와 일반 모드 간의 일관성을 테스트하는 함수
    
    Parameters
    ----------
    model_name : str _ 테스트할 모델의 이름
    """
    
    # CPU에서 테스트를 실행 (GPU가 있으면 'cuda'로 변경 가능)
    device = "cpu"
    
    # JIT 모드로 모델을 불러옴 (JIT는 PyTorch에서 모델을 최적화하는 방법)
    jit_model, transform = clip.load(model_name, device=device, jit=True)
    
    # 일반 모드로 동일한 모델을 불러옴 (JIT 없이)
    py_model, _ = clip.load(model_name, device=device, jit=False)

    # 테스트에 사용할 이미지를 불러옴 ("CLIP.png"라는 파일에서 이미지를 열고 변환)
    # 'transform'은 이미지를 모델에 맞는 형식으로 전처리
    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)
    
    # 세 개의 텍스트를 토큰화함 ("a diagram", "a dog", "a cat")
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # 네트워크 연산 중 그래디언트를 계산하지 않도록 설정 (테스트에서 불필요한 메모리 사용 방지)
    with torch.no_grad():
        # JIT 모드에서 이미지와 텍스트를 모델에 입력하여 결과를 얻음
        logits_per_image, _ = jit_model(image, text)
        # 결과에 대해 softmax를 적용하여 확률로 변환하고, CPU로 옮겨 numpy 배열로 변환
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # 동일한 이미지와 텍스트를 일반 모드의 모델에 입력하여 결과를 얻음
        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # JIT 모델과 일반 모델의 결과가 일정 범위 안에서 일치하는지 확인
    # 'np.allclose'는 두 배열이 허용 오차 범위 안에서 동일한지 확인하는 함수
    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)
    # atol: 절대 오차 허용 범위, rtol: 상대 오차 허용 범위
