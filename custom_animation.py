import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
rc('animation', html='html5')

import cv2 #put annotations

def play_one_episode(env, model, seed=None, n_max_steps=400):
    # Play 1 episode to create the frames
    frames = []
    rewards = [0]
    env.seed(seed)
    obs = env.reset()
    frames.append(env.render(mode="rgb_array"))
    for step in range(n_max_steps):
      action = model.predict(obs)
      obs, reward, done, info = env.step(action)
      frames.append(env.render(mode="rgb_array"))
      rewards.append(reward)
      if done:
        break
    env.close()
    return frames, rewards

def create_animation(frames, rewards, interval=30, position=(10,380), size=1.2, weight=1):
  fig, ax = plt.subplots()
  plt.axis('off')
  # Generate annotated frames
  reward = 0
  a_frames =[]
  for t in range(len(rewards)):
    reward += rewards[t]
    annotation = ("timestep: %3d, reward:"% (t)) 
    if rewards[t] >= 0:
      annotation = annotation + "+"
    else:
      annotation = annotation + "-"
    annotation = annotation + ("%.2f, total reward: %.2f" 
                          % (abs(rewards[t]), reward))
    a_frames.append(cv2.putText(frames[t], annotation, position, cv2.FONT_HERSHEY_PLAIN, size, (0, 0, 0), weight))

  patch = plt.imshow(a_frames[0])
  # Define a function to update the frame
  def animation(t):
    patch.set_data(a_frames[t])
    return patch
  anim = FuncAnimation(fig, animation, frames = len(rewards), 
                         interval = interval)
  return anim

def animate_one_episode(env, model, seed=None, n_max_steps=400, interval=30, position=(10,380), size=1.2, weight=1):
  frames, rewards = play_one_episode(env, model, seed=seed, n_max_steps=n_max_steps)
  return create_animation(frames, rewards, interval, position, size, weight)