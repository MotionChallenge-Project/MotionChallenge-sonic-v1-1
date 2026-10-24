"""Compile pinned author gatherers; compare observation, ONNX and motor targets.
Usage: python validate_sonic_native.py ADAPTER UPSTREAM DECODER ENCODER
Requires a C++17 compiler; inference uses the adapter's locked environment.
"""
from pathlib import Path
import subprocess, tempfile, json, importlib.util, sys, hashlib, re
import numpy as np
from scipy.spatial.transform import Rotation as R
root, source, decoder, encoder = map(lambda x:Path(x).resolve(),sys.argv[1:5])
sys.path.insert(0,str(root/'MotionChallenge-Core/src'))
from motion_challenge.sdk import StatePacket,ReferenceFrame
from motion_challenge.adapters.onnx import vector

native=source/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
main=(native/'src/g1_deploy_onnx_ref.cpp').read_text()
logger=(native/'src/state_logger.cpp').read_text()
header=(native/'include/state_logger.hpp').read_text()
def extract(text, marker):
 start=text.index(marker)
 a=text.index('{',start) if marker.startswith('struct ') else start+re.search(r'\)\s*(?:const\s*)?\{',text[start:]).end()-1
 depth=1;i=a+1
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i]
functions=['GatherMotionJointPositionsMultiFrame','GatherMotionJointVelocitiesMultiFrame','GatherMotionAnchorOrientationMutiFrame','GatherHisBodyJointPositions','GatherHisBodyJointVelocities','GatherHisLastActions','GatherHisBaseAngularVelocity','GatherHisGravityDir']
gather='\n'.join(extract(main,'bool '+name+'(') for name in functions)
entry=extract(header,'struct Entry {')+';'
latest=extract(logger,'std::vector<Entry> StateLogger::GetLatest(size_t n, bool').replace('StateLogger::','').replace('bool newest_first)','bool newest_first = true)')
timed=extract(logger,'std::vector<Entry> StateLogger::GetLatest(size_t n, double').replace('StateLogger::','').replace('bool newest_first)','bool newest_first = true)')
zero=extract(logger,'Entry StateLogger::makeZeroEntry_()').replace('StateLogger::','')
cpp='''#include <vector>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <memory>
#include <string>
#include "policy_parameters.hpp"
#include "math_utils.hpp"
class StateLogger { public:
'''+entry+'''
 mutable std::mutex ring_mutex_;
 size_t start_=0,size_=0,capacity_=1000;
 int configured_num_joints_=29,configured_num_actions_=29;
 double dt=.02;
 std::vector<Entry> ring_=std::vector<Entry>(capacity_);
'''+latest+'\n'+timed+'\n'+zero+'''
};
struct Motion {
 size_t timesteps=10;
 std::array<std::array<double,29>,10> q,dq;
 std::array<std::array<std::array<double,4>,1>,10> quat;
 size_t GetNumJoints(){return 29;} size_t GetNumBodyQuaternions(){return 1;}
 const double* JointPositions(int i){return q[i].data();}
 const double* JointVelocities(int i){return dq[i].data();}
 auto BodyQuaternions(int i){return quat[i];}
};
class Native {public:
 std::shared_ptr<StateLogger> state_logger_=std::make_shared<StateLogger>();
 std::shared_ptr<Motion> current_motion_=std::make_shared<Motion>();
 double control_dt_=.02,current_frame_=0;
 int saved_frame_for_observation_window_=0;
 bool has_upper_body_data_=false;
 std::array<double,17> upper_body_joint_positions_buffer_{},upper_body_joint_velocities_buffer_{};
 struct {bool play=true;} operator_state;
 std::array<double,4> init_actual{},init_ref{};
 std::array<double,4> ComputeApplyDeltaHeading(){return quat_mul_d(calc_heading_quat_d(init_actual),calc_heading_quat_inv_d(init_ref));}
'''+gather+'''
};
int main(){
 std::cout<<std::setprecision(17); Native n;
 for(auto kind:{0,1,2,3})for(int i=0;i<29;++i){int j=mujoco_to_isaaclab[i];double x=kind==0?default_angles[j]:kind==1?kps[j]:kind==2?kds[j]:g1_action_scale[j];std::cout<<x<<" ";}std::cout<<std::endl;
 int reset,mode,dimension;
 while(std::cin>>reset>>mode>>dimension){
  StateLogger::Entry entry; std::array<double,29> q,dq,last;
  for(auto &v:q)std::cin>>v;for(auto &v:dq)std::cin>>v;for(auto &v:last)std::cin>>v;
  for(auto &v:entry.base_quat)std::cin>>v;for(auto &v:entry.base_ang_vel)std::cin>>v;
  for(int i=0;i<10;++i){for(auto &v:n.current_motion_->q[i])std::cin>>v;for(auto &v:n.current_motion_->dq[i])std::cin>>v;for(auto &v:n.current_motion_->quat[i][0])std::cin>>v;}
  if(reset){n.state_logger_->size_=0;n.init_actual=entry.base_quat;n.init_ref=n.current_motion_->quat[0][0];}
  for(int i=0;i<29;++i){int j=mujoco_to_isaaclab[i];entry.body_q.push_back(q[j]-default_angles[j]);entry.body_dq.push_back(dq[j]);entry.last_action.push_back(last[i]);}
  n.state_logger_->ring_[n.state_logger_->size_++]=entry;
  std::vector<double> enc(dimension,0),history(930,0);
  n.GatherMotionJointPositionsMultiFrame(enc,4,10,1);
  n.GatherMotionJointVelocitiesMultiFrame(enc,294,10,1);
  n.GatherMotionAnchorOrientationMutiFrame(enc,584,10,1,mode);
  n.GatherHisBaseAngularVelocity(history,0,10,1);
  n.GatherHisBodyJointPositions(history,30,10,1);
  n.GatherHisBodyJointVelocities(history,320,10,1);
  n.GatherHisLastActions(history,610,10,1);
  n.GatherHisGravityDir(history,900,10,1);
  for(double v:enc)std::cout<<v<<" ";for(double v:history)std::cout<<v<<" ";std::cout<<std::endl;
  std::array<double,29> actions;for(auto &v:actions)std::cin>>v;
  for(int i=0;i<29;++i)std::cout<<static_cast<float>(default_angles[i]+actions[isaaclab_to_mujoco[i]]*g1_action_scale[i])<<" ";std::cout<<std::endl;
 }
}
'''
# This harness uses extracted author gatherers and logger padding verbatim;
# fixture transport supplies already sampled references, not a new data loader.
spec=importlib.util.spec_from_file_location('tested_adapter',root/'adapter.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
runtime=json.loads((root/'runtime.json').read_text());runtime['weights']=str(decoder);runtime['artifacts']['encoder']['path']=str(encoder)
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==runtime['source_commit'], 'upstream commit mismatch'
for used in [native/'src/g1_deploy_onnx_ref.cpp',native/'src/state_logger.cpp',native/'include/state_logger.hpp',native/'include/math_utils.hpp',native/'include/policy_parameters.hpp']:
 record=subprocess.check_output(['git','-C',str(source),'ls-tree','HEAD',str(used.relative_to(source))],text=True).split()
 blob=subprocess.check_output(['git','hash-object',str(used)],text=True).strip()
 assert record[2]==blob, f'upstream file changed: {used.name}'
policy=mod.SONIC(root,runtime);cfg=policy.cfg
canonical=tuple(json.loads((root/'MotionChallenge-Core/suites/lafan/dataset.json').read_text())['joint_names'])
order=[canonical.index(n) for n in policy.names]
rng=np.random.default_rng(1602);worst=dict(encoder_observation=0.,history=0.,decoder_input=0.,action=0.,motor_target=0.)
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory);(path/'parity.cpp').write_text(cpp)
 subprocess.run(['c++','-std=c++17','-Wno-narrowing','-I'+str(native/'include'),str(path/'parity.cpp'),'-o',str(path/'parity')],check=True)
 process=subprocess.Popen([str(path/'parity')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
 params=np.fromstring(process.stdout.readline(),sep=' ').reshape(4,29)
 for arr,key in zip(params,['default_joint_pos','joint_kp','joint_kd','action_scale']):np.testing.assert_array_equal(arr,vector(cfg[key],policy.names))
 for i in range(160):
  reset=i%40==0;actual=R.random(random_state=rng)
  q=rng.normal(0,.2,29);qd=rng.normal(0,.5,29);omega=rng.normal(0,.3,3)
  state=StatePacket(i*.02,canonical,tuple(q),tuple(qd),(0,0,.8),tuple(actual.as_quat()),(0,0,0),tuple(actual.apply(omega)))
  references=[]
  for offset in runtime['reference_offsets_s']:
   rq=rng.normal(0,.2,29);rv=rng.normal(0,.5,29);rr=R.random(random_state=rng)
   references.append(ReferenceFrame(i*.02+offset,i*.02+offset,tuple(rq),(0,0,.8),tuple(rr.as_quat()),tuple(rv)))
  # Repeated clamped endpoint frames exercise the reference-window boundary.
  if i%40>=35:references[2:]=[references[1]]*8
  references=tuple(references)
  if reset:policy.reset(state,references,1602)
  packed=np.concatenate([q,qd,policy.last,actual.as_quat()[[3,0,1,2]],omega]+[np.r_[np.array(r.joint_position_rad)[order],np.array(r.joint_velocity_rad_s)[order],np.array(r.root_orientation_xyzw)[[3,0,1,2]]] for r in references])
  dimension=cfg['encoder_dimension'];mode=int(cfg['reference_orientation_frame']=='heading')
  process.stdin.write(f'{int(reset)} {mode} {dimension} '+' '.join(map(repr,packed.tolist()))+'\n');process.stdin.flush()
  expected=np.fromstring(process.stdout.readline(),sep=' ');assert expected.shape==(dimension+930,),expected.shape
  expected=expected.astype(np.float32)
  encoder_actual=policy.encoder_observations(state,references)
  inputs=policy.observations(state,references)
  token=policy.encoder.run(['encoded_tokens'],{'obs_dict':expected[:dimension][None]})[0]
  native_input=np.r_[token.reshape(-1),expected[dimension:]][None]
  native_action=policy.session.run(['action'],{'obs_dict':native_input})[0].reshape(-1)
  actual_action=policy.infer(inputs)['action'].reshape(-1)
  process.stdin.write(' '.join(map(repr,native_action.tolist()))+'\n');process.stdin.flush()
  native_target=np.fromstring(process.stdout.readline(),sep=' ')
  observation_method=policy.observations;policy.observations=lambda *_:inputs
  command=policy.act(state,references);policy.observations=observation_method
  pairs=[('encoder_observation',encoder_actual[0],expected[:dimension]),('history',inputs['obs_dict'][0,64:],expected[dimension:]),('decoder_input',inputs['obs_dict'],native_input),('action',actual_action,native_action),('motor_target',command.joint_position_target_rad,native_target)]
  for name,a,b in pairs:
   worst[name]=max(worst[name],float(np.max(np.abs(np.asarray(a)-np.asarray(b)))))
   np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5,err_msg=name)
 process.stdin.close();assert process.wait(timeout=5)==0,process.stderr.read()
print(json.dumps(dict(cases=160,resets=4,source_commit=runtime['source_commit'],max_abs=worst,scope='Pinned C++ observation gatherers and StateLogger history/zero-padding, same author ONNX encoder and decoder, native float motor targets; canonical benchmark physics and CPU-vs-TensorRT numerical behavior are separate.')))
