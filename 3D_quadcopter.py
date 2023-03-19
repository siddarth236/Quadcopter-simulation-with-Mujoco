import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import matplotlib.pyplot as plt
import os
import control

xml_path = 'quadcopter_aarshit_manthan_siddarth.xml' #xml file (assumes this is in the same folder as this file)
simend = 10 #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

np.random.seed(1)
l = 0.8
mtot = 1.00
g = 9.81
Ix = 0.3
Iy = 0.3
Iz = 0.3
pos_lim = 0.1
phi_lim = 0.1
C = np.zeros((6,12))
C[0][0] = 1
C[1][1] = 1
C[2][2] = 1
C[3][9] = 1
C[4][10] = 1
C[5][11] = 1

#1) Set flag_estimator = 0 and test LQR controller
#2) Set flag_estimator = 1 and test LQE/Kalman filter
flag_track = 1 # 0 for hovering, 1 for trajectory tracking
flag_estimator = 0
a_dev = 5
alpha_dev = 20
pos_dev = 0.1
rate_dev = 0.1

def sin(theta):
    return np.sin(theta)

def cos(theta):
    return np.cos(theta)

def measurement():
    x_measured = data.qpos[0]+np.random.normal(0,pos_dev)
    y_measured = data.qpos[1]+np.random.normal(0,pos_dev)
    z_measured = data.qpos[2]+np.random.normal(0,pos_dev)
    phidot_measured = data.qvel[3]+np.random.normal(0,rate_dev)
    thetadot_measured = data.qvel[4]+np.random.normal(0,rate_dev)
    psidot_measured = data.qvel[5]+np.random.normal(0,rate_dev)
    y = np.array([x_measured,y_measured,z_measured,phidot_measured,thetadot_measured,psidot_measured]).reshape(-1,1)
    return y

def initialize(model,data):
    #pass
    global K,L,A,B

    A=np.array([[0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,-g,0,0,0,0,0,0,0],
                [0,0,0,g,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]])

    B=np.array([[0,0,0,0,0,0,0,0,1/mtot,0,0,0],
                [0,0,0,0,0,0,0,0,0,1/Ix,0,0],
                [0,0,0,0,0,0,0,0,0,0,1/Iy,0],
                [0,0,0,0,0,0,0,0,0,0,0,1/Iz]]).T
    Q = np.diag([18,25,25,1,1,1,0.09,0.09,0.16,1/2,1/2,1/2])
    #Q = np.eye(12)
    #R = np.array([[1/3600,0],[0,1/900]])*5
    R = np.diag([1/10000,1/2000,1/2000,1/2000])*1
    K,P,E=control.lqr(A,B,Q,R)
    #print(K.shape,P.shape,E.shape)
    
    Qe = np.diag([10000,10000,10000,25000,24000,24000])
    Re = np.eye(6)*0.01
    G=np.concatenate((np.zeros((6,6)),np.eye(6)),axis=0)
    L,P,E=control.lqe(A,G,C,Qe,Re)


def estimator(model,data,x_estimate,u):

    global L,A,B
    y = measurement()

    y_cap=np.matmul(C,x_estimate)
    x_capdot=np.matmul(A,x_estimate)+np.matmul(B,u).reshape(-1,1)+np.matmul(L,(y-y_cap))
    x_estimate=x_estimate+x_capdot*0.0001
    return x_estimate

def controller(model, data,state,state_ref):

    global K
    error=state_ref.reshape(-1,1)-state.reshape(-1,1)
    
    u1 = mtot*g
    u2 = 0
    u3 = 0
    u4 = 0

    del_u=np.matmul(K,error)
    u = np.array([u1,u2,u3,u4])
    u = u + del_u.reshape(-1)

    ## limiting force of motor
    for j,i in enumerate(u): 
        if i>60:
            u[j]=60
        if i<-60: 
            u[j]=-60

    fx_d = np.random.normal(0,a_dev)
    fy_d = np.random.normal(0,a_dev)
    fz_d = np.random.normal(0,a_dev)
    taux_d = np.random.normal(0,alpha_dev)
    tauy_d = np.random.normal(0,alpha_dev)
    tauz_d = np.random.normal(0,alpha_dev)
    data.qfrc_applied[0] = fx_d
    data.qfrc_applied[1] = fy_d
    data.qfrc_applied[2] = fz_d
    data.qfrc_applied[3] = taux_d
    data.qfrc_applied[4] = tauy_d
    data.qfrc_applied[5] = tauz_d

    phi = data.qpos[3]
    theta = data.qpos[4]
    psi = data.qpos[5]
    body = 1 #1 is bicopter (0 is world)
    
    data.xfrc_applied[body][0] = -(u[0])*(sin(phi)*sin(psi)+cos(phi)*cos(psi)*sin(theta))
    data.xfrc_applied[body][1] = -(u[0])*(cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi))
    data.xfrc_applied[body][2] = u[0]*cos(psi)*cos(theta)
    data.xfrc_applied[body][3] = u[1]
    data.xfrc_applied[body][4] = u[2]
    data.xfrc_applied[body][5] = u[3]
    
    return u

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = 90 ; cam.elevation = 0 ; cam.distance =  6.035328453847003
cam.lookat =np.array([ 0.0 , 0 , -2*flag_track])

x0_l = 0
y0_l = 0
z0_l = -2
# time and tspan
h = 0.0001
t0 = 0
tN = simend
pi = np.pi
T = simend
a = 2
b = 1

time = []
x = []; x_r=[]; xdot = []
y = []; y_r=[]; ydot = []
z = []; z_r=[]; zdot = []
phi = []; phidot = []
theta = []; thetadot = []
psi = []; psidot =[]
u1 = []; u2 = []
u3 = []; u4 = []

#initialize the controller
initialize(model,data)

x_estimate = np.zeros((12,1))
u = np.array([mtot*g,0,0,0])
while not glfw.window_should_close(window):
    time_prev = data.time
    t=data.time
    AA=2
    BB=AA
    tau_ref = 2*pi*(-15*(t/T)**4+6*(t/T)**5+10*(t/T)**3)
    taudot_ref = 2*pi*(-15*4*(1/T)*(t/T)**3+6*5*(1/T)*(t/T)**4+10*3*(1/T)*(t/T)**2)
    tauddot_ref = 2*pi*(-15*4*3*(1/T)**2*(t/T)**2 + 6*5*4*(1/T)**2*(t/T)**3+10*3*2*(1/T)**2*(t/T))

    x_ref = x0_l+AA*sin(a*tau_ref)
    z_ref = z0_l+BB*cos(b*tau_ref)
    xdot_ref =  AA*a*cos(a*tau_ref)*taudot_ref
    zdot_ref = -BB*b*sin(b*tau_ref)*taudot_ref
    xddot_ref = -AA*a*a*sin(a*tau_ref)*taudot_ref+AA*a*cos(a*tau_ref)*tauddot_ref
    zddot_ref = -BB*b*b*sin(b*tau_ref)*taudot_ref-BB*b*sin(b*tau_ref)*tauddot_ref
    #print(A*sin(a*psi_ref),sin(a*psi_ref),a*psi_ref,"ref")
    psi_ref=0
    psidot_ref=0
    y_ref=0
    ydot_ref=0
    yddot_ref=0
    theta_ref=0
    thetadot_ref=0
    phi_ref=0
    phidot_ref=0

    while (data.time - time_prev < 1.0/90.0):
        if flag_track==1:
            state_ref=np.array([x_ref,y_ref,z_ref,phi_ref,theta_ref,psi_ref,
                                xdot_ref,ydot_ref,zdot_ref,phidot_ref,thetadot_ref,psidot_ref])
        else:
            state_ref=np.zeros((12))
        if (flag_estimator==1):
            x_estimate = estimator(model,data,x_estimate,u)
        else:
            x_estimate = np.array([data.qpos[0],data.qpos[1],data.qpos[2],data.qpos[3],data.qpos[4],data.qpos[5],\
                                   data.qvel[0],data.qvel[1],data.qvel[2],data.qvel[3],data.qvel[4],data.qvel[5]])
        u = controller(model,data,x_estimate,state_ref)
        mj.mj_step(model, data)

    time.append(data.time)
    x.append(data.qpos[0]);x_r.append(x_ref)
    y.append(data.qpos[1]);y_r.append(y_ref)
    z.append(data.qpos[2]);z_r.append(z_ref)
    phi.append(data.qpos[3])
    theta.append(data.qpos[4])
    psi.append(data.qpos[5])
    xdot.append(data.qvel[0])
    ydot.append(data.qvel[1])
    zdot.append(data.qvel[2])
    phidot.append(data.qvel[3])
    thetadot.append(data.qvel[4])
    psidot.append(data.qvel[5])
    u1.append(u[0])
    u2.append(u[1])
    u3.append(u[2])
    u4.append(u[3])

    if (data.time>=simend):
        if flag_track==0:
            plt.figure(1)
            plt.subplot(3,1,1)
            plt.plot(time,x,'r')
            plt.plot(time,pos_lim*np.ones(len(time)),'k-.')
            plt.plot(time,-pos_lim*np.ones(len(time)),'k-.')
            plt.ylabel("x")
            plt.subplot(3,1,2)
            plt.plot(time,y,'r')
            plt.plot(time,pos_lim*np.ones(len(time)),'k-.')
            plt.plot(time,-pos_lim*np.ones(len(time)),'k-.')
            plt.ylabel("y")
            plt.subplot(3,1,3)
            plt.plot(time,z,'r')
            plt.plot(time,pos_lim*np.ones(len(time)),'k-.');
            plt.plot(time,-pos_lim*np.ones(len(time)),'k-.');
            plt.ylabel("z")

            plt.figure(2)
            plt.subplot(3,1,1)
            plt.plot(time,phi,'r')
            plt.plot(time,phi_lim*np.ones(len(time)),'k-.')
            plt.plot(time,-phi_lim*np.ones(len(time)),'k-.')
            plt.ylabel("phi")
            plt.subplot(3,1,2)
            plt.plot(time,theta,'r')
            plt.plot(time,phi_lim*np.ones(len(time)),'k-.')
            plt.plot(time,-phi_lim*np.ones(len(time)),'k-.')
            plt.ylabel("theta")
            plt.subplot(3,1,3)
            plt.plot(time,psi,'r')
            plt.plot(time,phi_lim*np.ones(len(time)),'k-.');
            plt.plot(time,-phi_lim*np.ones(len(time)),'k-.');
            plt.ylabel("psi")

            plt.figure(3)
            plt.subplot(2,2,1)
            plt.plot(time,u1,'r')
            plt.ylabel("u1")
            plt.subplot(2,2,2)
            plt.plot(time,u2,'r')
            plt.ylabel("u2")
            plt.subplot(2,2,3)
            plt.plot(time,u3,'r')
            plt.ylabel("u3")
            plt.subplot(2,2,4)
            plt.plot(time,u4,'r')
            plt.ylabel("u4")
            # plt.show()
            plt.show(block=False)
            plt.pause(10)
            plt.close()
            break
    
        elif flag_track ==1:
            plt.figure(1)
            plt.plot(x,z,label="actual")
            plt.plot(x_r,z_r,'r--',label="desired")
            plt.ylabel("z")
            plt.xlabel("x")
            plt.legend()
            
            plt.figure(2)
            plt.subplot(2,2,1)
            plt.plot(time,u1,'r')
            plt.ylabel("u1")
            plt.subplot(2,2,2)
            plt.plot(time,u2,'r')
            plt.ylabel("u2")
            plt.subplot(2,2,3)
            plt.plot(time,u3,'r')
            plt.ylabel("u3")
            plt.subplot(2,2,4)
            plt.plot(time,u4,'r')
            plt.ylabel("u4")
            plt.show()
            break


    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    #cam.lookat[0] = data.qpos[0]
    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
