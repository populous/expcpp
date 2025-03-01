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

# 안정한 시간 스텝 계산
dt = min(dx**2 / (4 * alpha), 0.05)  # 안정성 조건 적용
t_max = 1.0  # 최대 시간
n_steps = int(t_max / dt)

# 초기 온도 및 구조 상태 설정
T = np.zeros((ny, nx))  # 온도 배열
T[ny//2, nx//2] = 100  # 중심에서 100도의 열이 시작됨

displacement_x = np.zeros((ny, nx