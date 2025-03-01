import numpy as np
import matplotlib.pyplot as plt

# 물리적 파라미터 설정
Lx, Ly = 10.0, 10.0  # 전체 영역 크기
nx, ny = 40, 40  # 격자 크기
dx, dy = Lx / nx, Ly / ny  # 격자 간격

# 열전도 계수 및 구조 물성
alpha = 0.01  # 열전도 계수
E = 200e9  # 탄성 계수
nu = 0.3  # 포아송 비율
alpha_thermal = 1e-5  # 열팽창 계수

# 초기 온도 및 구조 상태 설정
T = np.zeros((ny, nx))  # 온도 배열
T[ny//2, nx//2] = 100  # 중심에서 100도의 열이 시작됨

displacement = np.zeros((ny, nx))  # 변위 배열

# 경계 조건
T[0, :] = 0  # 상단 경계
T[-1, :] = 0  # 하단 경계
T[:, 0] = 0  # 왼쪽 경계
T[:, -1] = 0  # 오른쪽 경계

# 서브도메인 분할 (열전도, 구조)
subdomain_thermal = T[:, :nx//2]
subdomain_structural = T[:, nx//2:]

# 열전도 방정식의 라플라시안 계산
def laplacian(T):
    """열전도 방정식의 라플라시안 연산"""
    lap = np.zeros_like(T)
    lap[1:-1, 1:-1] = (
        T[2:, 1:-1] + T[:-2, 1:-1] +
        T[1:-1, 2:] + T[1:-1, :-2] -
        4 * T[1:-1, 1:-1]
    )
    return lap

# 구조 방정식의 변형 계산 (기본적인 선형 해석)
def calculate_displacement(T):
    """온도에 의한 구조물의 변형 계산"""
    return alpha_thermal * E * T  # 단순화된 선형 모델 (온도에 의한 변형)

# 라그랑지 멀티플라이어 계산 (SPE 적용)
def lagrange_multiplier(T1, T2, lambda_val):
    """경계에서 온도를 일치시키기 위한 라그랑지 멀티플라이어"""
    # SPE 적용을 위한 경계에서 값 맞추기
    return T1 + lambda_val * (T2 - T1)

# 시간 발전
dt = 0.1  # 시간 간격
t_max = 1.0  # 최대 시간
n_steps = int(t_max / dt)

lambda_val = 0.1  # 라그랑지 멀티플라이어 초기값

# 시간 단계별로 계산
for step in range(n_steps):
    # 열전도 문제 해결
    lap_thermal = laplacian(subdomain_thermal)
    subdomain_thermal = subdomain_thermal + alpha * lap_thermal * dt
    
    # 구조 문제 해결
    displacement = calculate_displacement(subdomain_thermal)
    
    # 라그랑지 멀티플라이어를 통한 경계 조건 맞추기
    subdomain_thermal[:, -1] = lagrange_multiplier(subdomain_thermal[:, -1], subdomain_structural[:, 0], lambda_val)
    
    # 결과 출력 (매 100 스텝마다 출력)
    if step % 100 == 0:
        plt.imshow(T, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Temperature Distribution at Step {step}')
        plt.show()

# 최종 온도 분포 출력
plt.imshow(T, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Final Temperature Distribution')
plt.show()

# 변형 결과 출력
plt.imshow(displacement, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Structural Displacement')
plt.show()