# 🧠 Neuro-Dynamic Navigation: Dendritic PAI for TurtleBot3

> **Submission for "Quality of Optimization" Category**
> *Hypothesis Proven: Trading Memory for Speed—A 2.3x acceleration in robotic learning using Dynamic Parameter Expansion.*

## 📌 Project Overview
Standard Deep Reinforcement Learning (DRL) agents in robotics often suffer from slow adaptation times and "catastrophic forgetting." This project implements a **Dendritic Plasticity Artificial Intelligence (PAI)** agent for the TurtleBot3.

Unlike static neural networks, our agent **dynamically grows new connections (dendrites)** during training to tackle difficult navigation scenarios, mimicking the structural plasticity of biological brains.

---

## 🚀 Key Results: The Efficiency Trade-off

We benchmarked our **PAI Agent** against the industry-standard **Double DQN** (baseline) in a complex obstacle environment.

### 🏆 Performance Metrics

| Metric | Baseline (Double DQN) | **Dendritic PAI (Ours)** | **Improvement** |
| :--- | :--- | :--- | :--- |
| **Mastery Speed** | 16 Episodes | **7 Episodes** | **2.3x Faster Learning** ⚡ |
| **Success Rate** | ~85% (Struggles at corners) | **100%** (Clean runs) | **+15% Reliability** |
| **Parameter Count** | ~6,080 (Static) | ~12,160 (Dynamic) | **+100% Growth** |

### 📉 Optimization Analysis (Why this matters)
We focused on the optimization metric of **Learning Efficiency**. Our results confirm the hypothesis that **"Memory is cheaper than Time."**

1.  **The Strategy:** We allowed the PAI agent to dynamically expand its parameter count by ~100% (utilizing available RAM).
2.  **The Payoff:** This additional "neural capacity" allowed the agent to map complex spatial relationships significantly faster, reducing the training episodes required for mastery by **more than 50%**.
3.  **Conclusion:** For embedded robotics, trading a small amount of memory for a massive gain in adaptation speed is the superior optimization strategy.

![Comparison Graph](results/comparison_plot.png)
*(Figure 1: The PAI Agent (Green) achieves high reward stability significantly faster than the Baseline (Orange) and Standard DQN (Red).)*

---

## 🎥 Demonstration

**[Insert Link to YouTube Video Here]**

*(Above: The PAI Agent navigating the complex obstacle course without collisions after only 7 episodes of training.)*

---

## 🛠️ Methodology & Architecture

The agent utilizes a hybrid **Main Module + Dendrite Module** topology:

1.  **Main Module:** A standard fully connected backbone that handles general navigation.
2.  **Dendrite Module:** A masking layer that dynamically activates or "grows" weights based on the difficulty of the current state input.
3.  **Sparse Training:** While the model grows, it uses a sparsity mask to ensure that only relevant connections are updated, maintaining computational efficiency during the forward pass.

---

## 📂 Repository Structure

This repository contains the standalone agent code. It requires a standard ROS2/TurtleBot3 simulation environment.

```text
neuro-dynamic-navigation/
│
├── README.md               # Project documentation
├── models/                 # Pre-trained model weights
│   ├── pai_agent.pth       # Our Optimized Dendritic Agent
│   └── baseline_agent.pth  # Standard Double DQN (for comparison)
│
├── src/                    # Source Code
│   ├── dqn_agent_pai.py    # (CORE) The PAI Agent logic
│   ├── dqn_agent_double.py # The Baseline Agent logic
│   └── dqn_environment.py  # The RL Environment wrapper
│
└── results/                # Data artifacts
    └── comparison_plot.png # Training performance graph
