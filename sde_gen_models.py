import numpy as np

x0=9.1686 #### x0=ln(S0)
y0=0.3632 #### y0=ln(v0)
alpha=-0.0050 
gamma=11.2374 
mu=0.0396
sigma=0.5675 
rho=-0.7776
tt=10  ### number of observations per interval 
dd=252 ### number of days
n= dd*tt ### number of observations taken in dd intervals 
dt_sim = float(1)/float(252) #### the one that works best

def heston_model_fts(n=n,dt_sim=dt_sim,x0=x0,y0=y0,
    alpha=alpha,gamma=gamma,mu=mu,sigma=sigma,rho=rho):
    dt=dt_sim
    v0=np.exp(y0)
    N=2*n 
    Yt=np.empty((N, 2))
    Yt[0,0] = x0
    Yt[0,1] = v0
    for ii in range(1,N-1): 
        s = Yt[ii-1,0] 
        v = Yt[ii-1,1]
        z1, z2 = np.random.normal(0, 1, 2)
        zv=z1
        zs=rho*z1+np.sqrt(1-rho**2)*z2
        Yt[ii,0]=s +(alpha-0.5*max(v,0))*dt+(max(v,0)*dt)**0.5*zs 
        Yt[ii,1]=v +gamma*(mu-max(v,0))*dt+sigma*(max(v,0)*dt)**0.5*zv
        
    yy1=np.diff(Yt[:,0]) # we are taking the difference between Xn and Xn-1
    yy=yy1[n:N] #### take value from n to N.
    index=range(0,len(yy)-1, tt)
    FTS=yy[index]
    return FTS

def heston_model_rs(n=n,dt_sim=dt_sim,x0=x0,y0=y0,
    alpha=alpha,gamma=gamma,mu=mu,sigma=sigma,rho=rho):
    dt=dt_sim
    v0=np.exp(y0)
    N=2*n 
    Yt=np.empty((N, 2))
    Yt[0,0] = x0
    Yt[0,1] = v0
    for ii in range(1,N-1): 
        s = Yt[ii-1,0] 
        v = Yt[ii-1,1]
        z1, z2 = np.random.normal(0, 1, 2)
        zv=z1
        zs=rho*z1+np.sqrt(1-rho**2)*z2
        Yt[ii,0]=s +(alpha-0.5*abs(v))*dt+(abs(v)*dt)**0.5*zs 
        Yt[ii,1]=v +gamma*(mu-abs(v))*dt+sigma*(abs(v)*dt)**0.5*zv
        
    yy1=np.diff(Yt[:,0]) # we are taking the difference between Xn and Xn-1
    yy=yy1[n:N] #### take value from n to N.
    index=range(0,len(yy)-1, tt)
    RS=yy[index]
    return RS