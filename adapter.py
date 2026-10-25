"""Author SONIC encoder/decoder deployment contract, G1 encoder mode 0."""
from motion_challenge.adapters.onnx import cpu_session
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController, yaw
from motion_challenge.validation import sha256_file


def heading(rotation):
    # Author calc_heading_quat_d uses the projected local forward axis.
    return R.from_rotvec([0., 0., float(yaw(rotation))])


class SONIC(ONNXController):
    raw_action_history = True

    def __init__(self, root, runtime):
        super().__init__(root, runtime)
        artifact = runtime['artifacts']['encoder']
        path = root / artifact['path']
        if sha256_file(path) != artifact['sha256']:
            raise ValueError('encoder checkpoint checksum mismatch')
        self.encoder = cpu_session(path, artifact['sha256'])
        self.encoder_dim = self.cfg['encoder_dimension']
        if self.encoder.get_inputs()[0].shape != [1, self.encoder_dim]:
            raise ValueError('encoder input differs from pinned author configuration')
        if self.session.get_inputs()[0].shape != [1, 994]:
            raise ValueError('decoder input differs from pinned author configuration')

    def frame(self, state):
        q, qd, omega, gravity, _ = self.proprio(state)
        return [omega, q-self.default, qd, self.last, gravity]

    def reset(self, state, reference, seed):
        super().reset(state, reference, seed)
        self.history = [np.zeros((10,n), np.float64) for n in (3,29,29,29,3)]
        # Native StateLogger pads missing history with a zero quaternion.
        # Its quat_rotate_d(0, [0,0,-1]) returns [0,0,+1]. Keep that cold-start
        # behavior rather than silently repeating the current observation.
        self.history[-1][:,2] = 1.
        self.align = heading(R.from_quat(state.root_orientation_xyzw))*heading(R.from_quat(reference[0].root_orientation_xyzw)).inv()

    def encoder_observations(self, state, reference):
        if len(reference) != 10:
            raise ValueError('SONIC requires ten declared reference samples')
        q = np.asarray([r.joint_position_rad for r in reference])[:,self.order]
        qd = np.asarray([r.joint_velocity_rad_s for r in reference])[:,self.order]
        actual = R.from_quat(state.root_orientation_xyzw)
        left = heading(actual) if self.cfg['reference_orientation_frame']=='heading' else actual
        rotations = R.from_quat([r.root_orientation_xyzw for r in reference])
        rot6 = (left.inv()*self.align*rotations).as_matrix()[:,:,:2]
        obs = np.zeros((1,self.encoder_dim), np.float32)
        # encoder_mode_4 is [mode_id,0,0,0]; mode 0 is G1, not a one-hot.
        obs[0,4:294] = q.reshape(-1)
        obs[0,294:584] = qd.reshape(-1)
        obs[0,584:644] = rot6.reshape(-1)
        # Other modality fields stay zero, as in GatherEncoderObservations.
        return obs

    def observations(self, state, reference):
        self.history = [np.concatenate([h[1:],np.asarray(v)[None]]) for h,v in zip(self.history,self.frame(state))]
        tokens = self.encoder.run(['encoded_tokens'], {'obs_dict':self.encoder_observations(state,reference)})[0]
        return {'obs_dict':np.concatenate([tokens.reshape(-1)]+[h.reshape(-1) for h in self.history]).astype(np.float32)[None]}

    def act(self, state, reference):
        command = super().act(state, reference)
        # Native CreatePolicyCommand writes float q_target motor commands.
        return replace(command, joint_position_target_rad=tuple(float(np.float32(x)) for x in command.joint_position_target_rad))

    def close(self):
        self.encoder = None
        super().close()


def create(root, runtime):
    return SONIC(root, runtime)
