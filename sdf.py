import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


state_size = 44
action_branches = 4
action_branche1 = [3,3,3,2]

load_model = False
train_mode = True

discount_factor = 0.99
learning_rate = 3e-4
n_step = 128
batch_size = 128
n_epoch = 3
_lambda = 0.95
epsilon = 0.2

run_step = 100000 if train_mode else 0
test_step = 10000

print_interval = 10
save_interval = 100

# 유니티 환경 경로
game = "Antony.x86_64"
os_name = platform.system()
if os_name == 'Linux':
    env_name = f"./{game}"

# 모델 저장 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/{game}/PPO/{date_time}"
load_path = f"./saved_models/{game}/PPO/your_saved_model_date"

# 연산 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ActorCritic(torch.nn.Module):
    def __init__(self, state_size, action_branche1):
        super(ActorCritic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)

        # 각 행동 분기를 위한 별도의 출력층
        self.action_heads = torch.nn.ModuleList([torch.nn.Linear(128, branch_size) for branch_size in action_branche1])

        # 상태 가치 예측을 위한 출력층
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 각 행동 분기에 대해 softmax를 적용한 확률 분포 얻기
        action_probs = [F.softmax(head(x), dim=-1) for head in self.action_heads]
        
        # 상태 가치 예측
        state_values = self.value_head(x)

        return action_probs, state_values


# PPOAgent 클래스 정의
class PPOAgent:
    def __init__(self):
        self.network = ActorCritic(state_size, action_branche1).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = list()
        self.writer = SummaryWriter(save_path)

        if load_model:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
            ####################################################

    # 행동 선택 함수
    def get_action(self, state, training=True):
        # training이 true일 때는 훈련 모드, false일 때는 평가 모드
        self.network.train(training)
    
        # 신경망에서 정책(probabilities)와 상태 가치(state value) 추출
        policies, _ = self.network(torch.FloatTensor(state).to(device)) #(1,3)(1,3)(1,3)(1,2)
        
        
        # 각 분기별로 행동을 선택
        actions = [torch.multinomial(policy, 1).cpu().numpy() for policy in policies]
    
        # 선택된 행동을 numpy 배열로 변환하고, (1, 4) 형태로 반환
        return np.array(actions).reshape(1, -1)
    

    # 경험을 메모리에 추가 
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 모델 학습 함수
    def train_model(self):
        #ActorCritic 훈련모드로 시작
        self.network.train()

        # axis 는 0이 디폴트여서 안적어도됨. test.ipynb보기
        state      = np.stack([m[0] for m in self.memory], axis=0)
        action     = np.stack([m[1] for m in self.memory], axis=0)
        reward     = np.stack([m[2] for m in self.memory], axis=0)
        next_state = np.stack([m[3] for m in self.memory], axis=0)
        done       = np.stack([m[4] for m in self.memory], axis=0)
        
        #memory배열 비우는 작업
        self.memory.clear()

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device), [state, action, reward, next_state, done])
        print(state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        #torch.Size([128, 44]) torch.Size([128, 4]) torch.Size([128]) torch.Size([128, 44]) torch.Size([128])
        #자동미분기능을 잠시 끄는 명령어
        with torch.no_grad():
            
            
            state = state.squeeze(1)
            action = action.squeeze(1)
            reward = reward.squeeze(1)
            next_state = next_state.squeeze(1)
            done = done.squeeze(1)
            
            #test.ipynb 보기 3번쨰
            
            oldpolicies , value = self.network(state)
            
            
            
            # oldpolicies에서 action에 해당하는 확률값들을 가져오는 과정 128*4
            prob_old = torch.stack([
            oldpolicies[j][range(action.size(0)), action[:, j].long()]
            for j in range(action.size(1))
            ], dim=1)
            
            
            _, next_value = self.network(next_state)
            
            reward = reward.unsqueeze(1)  # [128] -> [128, 1]
            done = done.unsqueeze(1)  # [128] -> [128, 1]
            #
            delta = reward + (1 - done) * discount_factor * next_value - value
            
            #delta 값 복사 다르게 정의(128*1)
            adv = delta.clone()
            
            #지금 adv가 (128,1)인데 nstep도 128 이니 단지 128,1 에서 1,128로 바뀐거임 4번째 test.ipynb참조
            #nstep과 batch사이즈 정수배거나 같아야 편함 map -> adv, done 순회
            adv, done = map(lambda x: x.view(n_step, -1).transpose(0,1).contiguous(), [adv, done])
            
            # 역순 -> 126, 125, 124...0
            for t in reversed(range(n_step-1)):
                # adv[:, t]는 1*1 의 텐서 done 원래 스칼라인데 1*128로 가능 test.ipynb 4번쨰 보기
                #시간에 따른 누적보상
                adv[:, t] += (1 - done[:, t]) * discount_factor * _lambda * adv[:, t+1]
            
            # 다시 2차원
            adv = adv.transpose(0,1).contiguous().view(-1, 1)
            
            ret = adv + value

        actor_losses, critic_losses = [], []
        idxs = np.arange(len(reward))
        for _ in range(n_epoch):
            
            #[0,1,2,....127,128] 배열을 막 섞음
            np.random.shuffle(idxs)
            
            #(0, 128, 128)
            for offset in range(0, len(reward), batch_size):
                #idx는 길이 128 인 배열 5번쨰 보기 test.ipynb
                idx = idxs[offset : offset + batch_size]

                #5번쨰
                _state, _action, _ret, _adv, _prob_old = map(lambda x: x[idx], [state, action, ret, adv, prob_old])
                
                policies, value = self.network(_state)
                
                #일부러 슬라이싱해서 i:i+1   2차원 텐서유지
                prob = torch.stack([policy.gather(1, _action[:, i:i+1].long()).squeeze(1) for i, policy in enumerate(policies)], dim=1)
                print(_prob_old.shape,prob.shape)
                #각각의 원소들끼리 나눔
                #만약 prob이 [2.0, 3.0, 4.0]이고 _prob_old가 [1.0, 1.5, 2.0]이라면,
                # ratio는 [2.0/1.0, 3.0/1.5, 4.0/2.0] 즉 [2.0, 2.0, 2.0]이 됩니다.
                #ratio가 1보다 크면 더 좋은 정책
                ratio = prob / (_prob_old + 1e-7)
                
                surr1 = ratio * _adv
                
                #torch.clamp -> 값 제한 0.8부터 1.2  갑작스러운 정책변환 금지
                surr2 = torch.clamp(ratio, min=1-epsilon, max=1+epsilon) * _adv
                
                #actor_loss 스칼라값   mean()평균낸거임
                actor_loss = -torch.min(surr1, surr2).mean()

                # 이 걸 왜하는진 모르겠지만 일단
                critic_loss = F.mse_loss(value, _ret).mean()

                total_loss = actor_loss + critic_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    def write_summary(self, score, actor_loss, critic_loss, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/actor_loss", actor_loss, step)
        self.writer.add_scalar("model/critic_loss", critic_loss, step)

# Main 함수
if __name__ == '__main__':
    #엔진 설정 및 환경 파라미터 불러오는 부분
    engine_configuration_channel = EngineConfigurationChannel()
    environment_parameters_channel = EnvironmentParametersChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel, environment_parameters_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
     
    
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=5.0) ###그래픽 품질

    # PPOagent 지정
    agent = PPOAgent()
    
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0
    for step in range(run_step + test_step):
        
        #훈련스텝이 끝났을 때
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
            engine_configuration_channel.set_configuration_parameters(time_scale=1.0) # 평가모드에서의 타임 스케일
        
        #dec, term 가져오기
        dec, term = env.get_steps(behavior_name)

        #관측 가져오기
        state = dec.obs[0] #1*44모양
        action = agent.get_action(state, train_mode)
        
        
        #행동 튜플로 변환 (1,4를 받아야함) [[2,1,0,1]] 각 브랜치에서 선택된 행동
        action_tuple = ActionTuple(discrete=action)
        print(action_tuple.discrete)
        env.set_actions(behavior_name, action_tuple)

        dec, term = env.get_steps(behavior_name)
        
        ##
        done = [False]
        print(f"done list length: {len(done)}")
        next_state = dec.obs[0] # 1*44
        reward = dec.reward
        
        # 여기서 reward 값을 출력하여 확인
        print(f"Step: {step}, Rewards: {reward}")
        
        # 단일이면 이렇게 안해도됨
        for id in term.agent_id:
            _id = list(term.agent_id).index(id)
            print(f"_id: {_id}")
            done[_id] = True
            next_state = term.obs[0] # 어차피 에이전트 하나여서 0으로 해도 될듯?
            reward = term.reward[_id]
        score += reward

        if train_mode:
            for id in range(len(dec.agent_id)):
                agent.append_sample(state, action, reward, next_state, done)

            if (step + 1) % n_step == 0:
                actor_loss, critic_loss = agent.train_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        if done[0]:
            episode += 1
            scores.append(score)
            score = 0

            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
                agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, step)
                actor_losses, critic_losses, scores = [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                      f"Actor loss: {mean_actor_loss:.2f} / Critic loss: {mean_critic_loss:.4f}")

            if train_mode and episode % save_interval == 0:
                agent.save_model()

    env.close()

