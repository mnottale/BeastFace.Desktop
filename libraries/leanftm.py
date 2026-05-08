import mediapipe
import numpy as np
import json
import cv2
import os

_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))

fph = None
ctb = None
htc = None
pcf = None

# lazy init
mesh_renderer = None 
faces_fixed = None
face_mesh = None
canon_ranges = list()
target_resolution = 256

shoot_faces = None # cache

lipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
lipsLowerInner = [95, 88, 178, 87, 14, 317, 402, 318, 324] #, 308] [78,

lipsOut = lipsUpperOuter + list(reversed(lipsLowerOuter))
lipsIn = lipsUpperInner + list(reversed(lipsLowerInner))

rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
rightEyebrowLower = [35, 124, 46, 53, 52, 65]
rightEyebrow = list(rightEyebrowUpper) + list(reversed(rightEyebrowLower))

leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
leftEyebrowLower = [265, 353, 276, 283, 282, 295]
leftEyebrow = leftEyebrowUpper + list(reversed(leftEyebrowLower))

rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]
rightEye0 = rightEyeUpper0 + list(reversed(rightEyeLower0))

rightEyeIris = [474, 475, 476, 477]

leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]

leftEye0 = leftEyeUpper0 + list(reversed(leftEyeLower0))

leftEyeIris = [469, 470, 471, 472]
muzletip = [393, 164, 167, 267, 0, 37]

important = set(muzletip + leftEyeIris + rightEyeIris + leftEye0 + rightEye0 + leftEyebrow + rightEyebrow + lipsOut + lipsIn)


def compute_4x4_transformation_matrix(V_ref, V_capt, weights=None):
    if weights is None:
        weights = np.ones(len(V_ref))  # Equal weights if not provided

    # Normalize weights
    weights = weights / np.sum(weights)

    # Weighted centroids
    mu_ref = np.sum(weights[:, None] * V_ref, axis=0)
    mu_capt = np.sum(weights[:, None] * V_capt, axis=0)

    # Center the vertices
    V_ref_centered = V_ref - mu_ref
    V_capt_centered = V_capt - mu_capt

    # Weighted cross-covariance matrix
    H = np.dot((weights[:, None] * V_capt_centered).T, V_ref_centered)

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)

    # Compute scaling factor
    ref_aligned = np.dot(V_ref_centered, R.T)
    scale = np.linalg.norm(weights[:, None] * V_capt_centered) / np.linalg.norm(weights[:, None] * ref_aligned)

    #R = modify_x_rotation(R, 0.001)
    R_scaled = scale * R

    # Compute translation
    T = mu_capt - np.dot(mu_ref, R_scaled)

    # Assemble the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_scaled
    transformation_matrix[:3, 3] = T

    return transformation_matrix

def load_obj(path):
    verts = list()
    faces = list()
    def xtract(s):
        return int(s.split('/')[0])-1
    with open(path, 'r') as fd:
        for l in fd.read().split('\n'):
            comps = l.split(' ')
            if comps[0] == 'v':
                verts.append((float(comps[1]), float(comps[2]), float(comps[3])))
            elif comps[0] == 'f':
                faces.append((xtract(comps[1]), xtract(comps[2]), xtract(comps[3])))
    return {'verts': verts, 'faces': faces}



from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

eye_inner = [385,386,387, 373,374,380,  158,159,160,144,145,153]

xaxis = None
yaxis = None
zaxis = None
xscale = None
def morph(landmarks):
    global xaxis, yaxis, zaxis, xscale
    xaxis = np.array(landmarks[338])-np.array(landmarks[109])
    xscale = np.linalg.norm(xaxis)
    xaxis /= xscale
    yaxis = np.array(landmarks[9])-np.array(landmarks[10])
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)
    zaxis /= np.linalg.norm(zaxis)

def proj(v):
    v = np.array(v)
    return np.array([np.dot(v, xaxis), np.dot(v, yaxis), np.dot(v, zaxis)])
class MeshRenderer:
    def __init__(self, width, height):
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(50, 50)
        glutCreateWindow(b"dummy glut")
        """Initialize the offscreen OpenGL rendering context."""
        self.width = width
        self.height = height

        # Create an offscreen framebuffer
        self.framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)

        # Create a texture to render into
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)

        # Create a renderbuffer for depth testing
        self.depthbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depthbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depthbuffer)

        # Check if framebuffer is complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise Exception("Framebuffer is not complete! {}".format(glCheckFramebufferStatus(GL_FRAMEBUFFER)))

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def render_mesh(self, vertices, faces):
        """Render the mesh to the framebuffer."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # Set up projection and view
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(22, self.width / self.height, 0.1, 100)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0.5, 0.5, -2, 0.5, 0.5, 0, 0, 1, 0)


        # Render the mesh
        cidx = 2
        minz = min(map(lambda p:p[cidx], vertices))
        maxz = max(map(lambda p:p[cidx], vertices))
        #print('zrange: {} {} {}'.format(minz, maxz, maxz-minz))
        glBegin(GL_TRIANGLES)
        for i, face in enumerate(faces):
            for vertex_index in face:
                z = vertices[vertex_index][cidx]
                c = max(0.5, 1.0-(z-minz)/(maxz-minz)*1.0) #/2.0 + 0.5
                cc = (c, c, c)
                has_color = False
                if vertex_index in important:
                    if vertex_index in eye_inner:
                        cc = (1.0,0.0,1.0)
                        has_color = True
                    elif vertex_index in leftEye0 or vertex_index in rightEye0:
                        cc = (1.0, 0.0, 0.0)
                        has_color = True
                    elif vertex_index in muzletip:
                        cc = (0.0, 1.0, 0.0)
                        has_color = True
                    #elif vertex_index in leftEyebrow or vertex_index in rightEyebrow:
                    #    cc = (0.0,0.0,1.0)
                    #elif vertex_index in lipsIn or vertex_index in lipsOut:
                    #    cc = (1.0,1.0,0.0)
                    elif vertex_index in leftEyeIris or vertex_index in rightEyeIris:
                        cc = (1.0, 0.0, 1.0)
                        has_color = True
                if i in red_faces:
                    cc = (0,0.0,max(0.0, c*3-2.0))
                    has_color = True
                if not has_color:
                    # Fixed texture
                    cverts = canon["verts"]
                    if vertex_index >= len(cverts):
                        cc = (0.5,0.5,0.5)
                    else:
                        vf = canon["verts"][vertex_index]
                        vals = list()
                        for idx in range(3):
                            val = (vf[idx]-canon_ranges[idx][0])/(canon_ranges[idx][1]-canon_ranges[idx][0])
                            vals.append(val)
                        cc = (abs(vals[0]-0.5)*2.0, 0.2 + vals[1]*0.6, 0.2)
                glColor3fv(cc)
                glVertex3fv(vertices[vertex_index])
        glEnd()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def get_rendered_image(self):
        """Retrieve the rendered image as a NumPy array."""
        glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Convert to a NumPy array and flip vertically (OpenGL origin is bottom-left)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = np.fliplr(image)
        return image

def morph2(verts, faces, cvlen, cflen):
    global morphdata, morphdate, red_faces, shoot_faces
    ears_base_vertex = [(590, 591, 592),  (595, 596, 597)]
    for side, (bi1, bim, bi2) in enumerate(ears_base_vertex):
        sgn = (side == 0) and 1 or -1
        bleft = np.array(verts[cvlen+bi1])
        bmid = np.array(verts[cvlen+bim])
        bright = np.array(verts[cvlen+bi2])
        bavg = (bleft+bright)/2.0
        vdir = bmid - bavg
        hdir = bright-bleft
        vdir *= 12.0
        hdir /= 4.0
        bhead1 = bmid + vdir * 1.0 + hdir *  1.0
        bhead2 = bmid + vdir * 1.0 - hdir *  1.0
        i = len(verts)
        verts.append(bhead1.tolist())
        verts.append(bhead2.tolist())
        verts.append(bleft.tolist())
        verts.append(bright.tolist())
        faces.append([i, i+1, i+2])
        faces.append([i, i+2, i+3])

    mcenter = (np.array(verts[0]) + np.array(verts[164])) /2
    mfactor = 0.0
    #print(mfactor)
    mend = np.array(verts[199])
    # we need to split x and ytop/bottom ranges as mouth opens and we want lower
    # jaw to be handled
    mextendlow = np.linalg.norm(proj(mend-mcenter)[0:2])*1.05
    mextendhi = np.linalg.norm(proj(np.array(verts[5])-mcenter)[0:2])*1.05
    factor_func = lambda dxr, dyr : 1.0 - (dxr**1.5)*1.0 - (dyr**1.5)*1.0
    for i, v in enumerate(verts[:468]):
        vv = np.array(v)
        delta = proj(vv-mcenter)
        dist = np.linalg.norm(delta[0:2])
        if dist > mextendlow:
            continue
        if delta[1] < 0 and dist > mextendhi:
            continue
        delta[2] = -xscale * mfactor * factor_func(abs(delta[0]/mextendlow), abs(delta[1]/mextendlow))
        newpoint = mcenter + delta[0]*xaxis + delta[1]*yaxis + delta[2]*zaxis
        verts[i] = newpoint
    for i in range(16):
        del faces[882]
    if shoot_faces is None:
        shoot_faces = list()
        edges = [(409,408,407, 291,306,292, 308, 415, 324,  410, 287,  325),
        (191,183, 184, 185, 186, 95, 96, 78, 62, 76, 61, 57)]
        i = 0
        while i < len(faces):
            f = faces[i]
            shoot = False
            for es in edges:
                count = 0
                for e in es:
                    if e in f:
                        count += 1
                if count > 2:
                    shoot = True
                    break
            if shoot:
                del faces[i]
                shoot_faces.append(i)
            else:
                i+=1
    else:
        for i in shoot_faces:
            del faces[i]
    red_faces = list()
    faces.append([13, 432, 212])
    red_faces.append(len(faces)-1)
    faces.append([14, 212, 432])
    red_faces.append(len(faces)-1)
def initialize_head_mesh():
    global canon, fph, cth, htc, faces_fixed, canon_ranges
    with open(os.path.join(_ASSET_DIR, 'canonical-face-mesh.json'), 'r') as fd:
        canon = json.loads(fd.read())
        for idx in range(3):
            vmin = min(map(lambda v:v[idx], canon["verts"]))
            vmax = max(map(lambda v:v[idx], canon["verts"]))
            canon_ranges.append((vmin, vmax))
    fph = load_obj(os.path.join(_ASSET_DIR, 'face-plus-head.obj'))
    cth = dict()
    htc = dict()
    thresh = 0.01
    for i, v in enumerate(fph['verts']):
        if abs(v[0]) > 20 or abs(v[1]) > 20 or abs(v[2]) > 20:
            print('bogus vert {} {}'.format(i, v))
        for i2, v2 in enumerate(canon['verts']):
            if (v[0]-v2[0])**2 + (v[1]-v2[1])**2 + (v[2]-v2[2])**2 < thresh*thresh:
                cth[i2] = i
                htc[i] = i2
                break
    print('matcher hit: {}/{}'.format(len(cth), len(canon['verts'])))
    delta = 478 # len(canon['verts']) # which is 468, wtf?
    faces_fixed = list()
    for (fa, fb, fc) in fph['faces']:
        if fa in htc and fb in htc and fc in htc:
            continue
        if fa in htc:
            fa = htc[fa]
        else:
            fa += delta
        if fb in htc:
            fb = htc[fb]
        else:
            fb += delta
        if fc in htc:
            fc = htc[fc]
        else:
            fc += delta
        faces_fixed.append((fa, fb, fc))
    return canon, fph, cth, htc

def render3d(landmarks, target_resolution):
    morph(landmarks)
    from custom.face_geometry import (  # isort:skip
        PCF,
        get_metric_landmarks,
        procrustes_landmark_basis,
        landmark_weights
    )
    global mesh_renderer, canon, pcf, custom_canon_verts
    if pcf is None:
        frame_width = 1 #target_resolution
        frame_height = 1 #target_resolution
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
          [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
          dtype="double",
        )
        pcf = PCF(
            near=1,
            far=10000,
            frame_height=frame_height,
            frame_width=frame_width,
            fy=camera_matrix[1, 1],
        )
    if mesh_renderer is None:
        initialize_head_mesh()
        mesh_renderer = MeshRenderer(target_resolution, target_resolution)
     # Do two steps, since the code has to estimate rotation and scale
    pose_transform_mat = compute_4x4_transformation_matrix(canon['verts'][:468], landmarks[:468], landmark_weights)
    points = np.array(canon['verts'][:468])
    N = points.shape[0]
    points_homogeneous = np.hstack([points, np.ones((N, 1))])
    points2 = points_homogeneous @ pose_transform_mat.T
    points2 = points2[:,:3]
    transform2 =  compute_4x4_transformation_matrix(points2, landmarks[:468], landmark_weights)
    #print(transform2)
    pose_transform_mat = np.dot(transform2, pose_transform_mat)
    v = canon['verts'][0]
    v = np.dot(pose_transform_mat,np.array([v[0], v[1], v[2], 1]))[0:3].tolist()
    #print('calc: {}  mes: {}'.format(v, landmarks[0]))
    factor = (1 + landmarks[0][2]) / (1+v[2])
    # FIXME crap python-iterating code, see above
    tverts = list()
    for v in fph['verts']:
         v = np.dot(pose_transform_mat,np.array([v[0], v[1], v[2], 1]))[0:3].tolist()
         #v = [(v[0]+1)*factor-1, (v[1]+1)*factor-1, (v[2]+1)*factor-1]
         #v = [v[0]*factor, v[1]*factor, v[2]*factor]
         #if len(tverts) == 0:
         #    print(v)
         tverts.append(v)
    llist = landmarks.tolist()
    rverts = llist + tverts
    rfaces = canon['faces'] + faces_fixed
    morph2(rverts, rfaces, len(llist), len(canon['faces']))
    #mesh_renderer.render_mesh(landmarks, canon['faces'])
    mesh_renderer.render_mesh(rverts, rfaces)
    #mesh_renderer.render_mesh(tverts, fph['faces'])
    # Get the rendered image
    return mesh_renderer.get_rendered_image()

def face_to_model(in_img, target_resolution, render_3d=True, rpmode=True):
    global face_mesh
    if face_mesh is None:
        face_mesh = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    image = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    for face_landmarks in results.multi_face_landmarks:
        landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
        break
    return render3d(landmarks, target_resolution)