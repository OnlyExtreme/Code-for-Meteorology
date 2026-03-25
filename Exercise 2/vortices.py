import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import dst, idst
from numba import njit, prange

# ==========================================
# 1. 核心数学求解器 [cite: 25, 45, 111]
# ==========================================
def PoiSin(ML, NL, D):
    M, N = ML - 2, NL - 2
    i = np.arange(1, M + 1)
    j = np.arange(1, N + 1)
    I, J = np.meshgrid(i, j, indexing='ij')
    # 对应文档公式：D^2 / (2 * (cos(...) + cos(...) - 2)) [cite: 42]
    return (D * D / 2) / (np.cos(I * np.pi / (M + 1)) + np.cos(J * np.pi / (N + 1)) - 2)

def Poisson(ML, NL, PoiSinCoe, vor):
    st = np.zeros((ML, NL))
    vot = vor[1:-1, 1:-1]
    # 使用 2D DST 求解泊松方程 [cite: 82, 97, 108]
    vordst = dst(dst(vot, type=1, axis=0, norm='ortho'), type=1, axis=1, norm='ortho')
    phidst = vordst * PoiSinCoe
    phidst = idst(idst(phidst, type=1, axis=1, norm='ortho'), type=1, axis=0, norm='ortho')
    st[1:-1, 1:-1] = phidst
    return st

@njit(parallel=True)
def ArakawaJ(A, B, D):
    # Arakawa Jacobian 保持能量和拟能守恒 [cite: 111, 113]
    ML, NL = A.shape
    JAB = np.zeros((ML, NL))
    dd12 = 1.0 / (12.0 * D * D)
    
    for ii in prange(1, ML - 1): # 对应 MATLAB 索引 2:ML-1 [cite: 116]
        for jj in range(1, NL - 1): # 对应 MATLAB 索引 2:NL-1 [cite: 119]
            ip, im = ii + 1, ii - 1
            jp, jm = jj + 1, jj - 1
            
            # 严格遵循文档中的 Arakawa 9点差分格式 [cite: 122-129]
            pj =  A[ip, jj] * (B[ii, jp] - B[ii, jm] + B[ip, jp] - B[ip, jm])
            pj += A[ip, jp] * (B[ii, jp] - B[ip, jj])
            pj += A[ii, jp] * (B[im, jj] - B[ip, jj] + B[im, jp] - B[ip, jp])
            pj += A[im, jp] * (B[im, jj] - B[ii, jp])
            pj += A[im, jj] * (B[ii, jm] - B[ii, jp] + B[im, jm] - B[im, jp])
            pj += A[im, jm] * (B[ii, jm] - B[im, jj])
            pj += A[ii, jm] * (B[ip, jj] - B[im, jj] + B[ip, jm] - B[im, jm])
            pj += A[ip, jm] * (B[ip, jj] - B[ii, jm])
            
            JAB[ii, jj] = pj * dd12

    
    return JAB

# ==========================================
# 2. 网格与碰撞初始条件 [cite: 8, 11, 14]
# ==========================================
Lx, Ly = 10.0, 10.0
Nx, Ny = 101, 101
D = Lx / (Nx - 1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# 四个涡旋中心 [cite: 11, 12]
xi_1 =  np.exp(-6*((X-3.3)**2+(Y-3)**2));  xi_2 = -np.exp(-6*((X-4.8)**2+(Y-3)**2))
xi_3 = -np.exp(-6*((X-5.2)**2+(Y-7)**2));  xi_4 =  np.exp(-6*((X-6.7)**2+(Y-7)**2))

vor = xi_1 + xi_2 + xi_3 + xi_4
poi_sin_coe = PoiSin(Nx, Ny, D)
psi_0 = Poisson(Nx, Ny, poi_sin_coe, vor)

# 启动步 [cite: 18]
dt = 0.015 
vor_old = vor.copy()
vor_curr = vor - dt * ArakawaJ(psi_0, vor, D)
psi_curr = Poisson(Nx, Ny, poi_sin_coe, vor_curr)

# ==========================================
# 3. 动画设置 (速度场)
# ==========================================
fig, ax = plt.subplots(figsize=(8, 7))

def get_velocity(psi):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    # u = -d(psi)/dy, v = d(psi)/dx [cite: 6]
    u[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * D)
    v[1:-1, 1:-1] =  (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * D)
    return u, v

def animate(frame):
    global vor_old, vor_curr, psi_curr
    
    # 模拟推进 [cite: 20, 23]
    for _ in range(60):
        J = ArakawaJ(psi_curr, vor_curr, D)
        vor_next = vor_old - 2 * dt * J
        vor_next[0,:]=vor_next[-1,:]=vor_next[:,0]=vor_next[:,-1]=0
        psi_next = Poisson(Nx, Ny, poi_sin_coe, vor_next)
        
        vor_old, vor_curr, psi_curr = vor_curr, vor_next, psi_next
    
    u, v = get_velocity(psi_curr)
    speed = np.sqrt(u**2 + v**2)
    
    ax.clear()
    # 背景显示流速大小
    cf = ax.pcolormesh(X, Y, speed, cmap='Reds', shading='auto', vmin=0, vmax=0.15)
    # 绘制动态流线
    ax.streamplot(X.T, Y.T, u.T, v.T, color='white', linewidth=0.8, density=1.2, arrowsize=0.8)
    
    ax.set_title(f"Velocity Field Evolution | Time: {frame*4*dt:.2f}")
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
    ax.set_aspect('equal')

ani = animation.FuncAnimation(fig, animate, frames=300, interval=15)
plt.show()